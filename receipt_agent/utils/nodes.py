import base64
import json

from langchain.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from receipt_agent.utils.state import ItemizationBreakdown, ReceiptData, ReceiptState


def load_image_node(state: ReceiptState):
    messages = state.get("messages", [])
    if not messages:
        return {"image_bytes": None}

    last_message = messages[-1]

    if isinstance(last_message.content, list):
        for content_block in last_message.content:
            if isinstance(content_block, dict) and content_block.get("type") == "image":
                base64_data = content_block.get("data")

                if base64_data:
                    try:
                        image_bytes = base64.b64decode(base64_data)
                        return {"image_bytes": image_bytes}
                    except Exception as e:
                        print(f"Error decoding base64: {e}")
                        return {"image_bytes": None}

    return {"image_bytes": None}


def analyze_image_node(state: ReceiptState):
    image_base64 = base64.standard_b64encode(state["image_bytes"]).decode("utf-8")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_model = model.with_structured_output(ReceiptData)

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": """Extract all items from this receipt with their individual prices,
                plus the tax amount and total. Also give an analysis of the receipt, including the total number of items,
                the total amount, and the total tax. Be concise and to the point.""",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ]
    )

    receipt_data = structured_model.invoke([message])

    return {
        "analysis": receipt_data.analysis,
        "receipt_items": [item.dict() for item in receipt_data.items],
        "total_tax": receipt_data.tax,
    }


def ask_split_node(state: ReceiptState):
    user_input = interrupt(
        {
            "type": "form",
            "fields": [
                {
                    "name": "num_people",
                    "label": "How many people do you want to split the bill with?",
                    "type": "number",
                    "required": True,
                },
                {"name": "people_names", "label": "Enter the names (comma-separated)", "type": "text", "required": True},
            ],
        }
    )

    num_people = int(user_input.get("num_people"))
    people_names_input = user_input.get("people_names")

    if isinstance(people_names_input, list):
        people_names = people_names_input
    else:
        people_names = [name.strip() for name in people_names_input.split(",")]
    ai_message = AIMessage(content=f"Got it! Splitting the bill with {num_people} people: {', '.join(people_names)}")

    return {"num_people": num_people, "people_names": people_names, "messages": [ai_message]}


def get_split_instructions_node(state: ReceiptState):
    """Ask user how to split the bill"""

    items = state.get("receipt_items", [])
    people_names = state.get("people_names", [])

    # Build a nice prompt listing all items
    items_list = "\n".join([f"{i+1}. {item['name']}: ${item['price']:.2f}" for i, item in enumerate(items)])
    people_list = ", ".join(people_names)

    # interrupt() returns the value directly, not a dict
    split_method = interrupt(
        f"""How should we split the bill?

Items:
{items_list}

People: {people_list}

Examples:
- "Item 1 split between Alice and Bob"
- "Item 2 Alice pays"
- "Item 3 all of us"
- "Item 4 2 shares Alice, 1 share Bob"
- "Split everything equally"
- "I pay for items 1-2, they split items 3-4" """
    )

    # split_method is now a string, not a dict
    ai_message = AIMessage(content=f"Processing: {split_method}")

    return {"split_instructions": {"raw": split_method}, "messages": [ai_message]}


def calculate_itemization_node(state: ReceiptState):
    """Parse split instructions and calculate who owes what"""

    items = state.get("receipt_items", [])
    people_names = state.get("people_names", [])
    split_instructions = state.get("split_instructions", {})
    total_tax = state.get("total_tax", 0.0)

    # Create model with structured output
    model = ChatOpenAI(model="gpt-5-mini", temperature=0)
    structured_model = model.with_structured_output(ItemizationBreakdown)

    items_str = json.dumps(items)
    people_str = json.dumps(people_names)

    prompt = f"""Parse these split instructions and calculate how much each person owes for each item.

Items: {items_str}
People: {people_str}
Split Instructions: {split_instructions.get('raw', '')}

Calculate the breakdown of who owes what for each item, including:
- Each item with its name, price, and how much each person owes for it
- The total each person owes before tax
- The overall total before tax"""

    breakdown = structured_model.invoke([HumanMessage(content=prompt)])

    item_breakdown_dicts = []
    for item in breakdown.item_breakdown:
        splits_dict = {split.person_name: split.amount for split in item.splits}
        item_breakdown_dicts.append({"item_name": item.item_name, "item_price": item.item_price, "splits": splits_dict})

    person_totals_dict = {person.person_name: person.total for person in breakdown.person_totals}

    breakdown_dict = {
        "item_breakdown": item_breakdown_dicts,
        "person_totals": {},
        "total_before_tax": breakdown.total_before_tax,
        "total_tax": total_tax,
    }

    # Add tax proportionally to each person
    subtotal = breakdown.total_before_tax
    for person, person_subtotal in person_totals_dict.items():
        if subtotal > 0:
            tax_share = total_tax * (person_subtotal / subtotal)
            breakdown_dict["person_totals"][person] = person_subtotal + tax_share
        else:
            breakdown_dict["person_totals"][person] = person_subtotal

    ai_message = HumanMessage(content=f"Split calculated:\n{json.dumps(breakdown_dict, indent=2)}")

    return {"person_itemization": breakdown_dict, "messages": [ai_message]}
