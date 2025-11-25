from typing import NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ReceiptState(MessagesState):
    """State of the receipt agent"""

    image_bytes: NotRequired[bytes]
    analysis: NotRequired[str]
    num_people: NotRequired[int]
    people_names: NotRequired[list[str]]
    receipt_items: NotRequired[list[dict]]
    total_tax: NotRequired[float]
    split_instructions: NotRequired[dict]
    person_itemization: NotRequired[dict]


class ReceiptItem(BaseModel):
    """An item on the receipt"""

    name: str = Field(description="Name of the item")
    price: float = Field(description="Price of the item")


class ReceiptData(BaseModel):
    """Data extracted from the receipt"""

    items: list[ReceiptItem] = Field(description="List of items on the receipt")
    subtotal: float = Field(description="Subtotal of the receipt")
    tax: float = Field(description="Tax amount of the receipt")
    total: float = Field(description="Total amount of the receipt")
    analysis: str = Field(description="Analysis of the receipt")


class PersonSplit(BaseModel):
    """How much a person owes for an item"""

    person_name: str = Field(description="Name of the person")
    amount: float = Field(description="Amount this person owes for the item")


class PersonTotal(BaseModel):
    """Total amount a person owes"""

    person_name: str = Field(description="Name of the person")
    total: float = Field(description="Total amount this person owes before tax")


class ItemBreakdown(BaseModel):
    """Structure for each item in the breakdown"""

    item_name: str = Field(description="Name of the item")
    item_price: float = Field(description="Price of the item")
    splits: list[PersonSplit] = Field(description="Dictionary mapping person name to amount they owe for this item")


class ItemizationBreakdown(BaseModel):
    """Breakdown of who owes what for each item"""

    item_breakdown: list[ItemBreakdown] = Field(description="List of items with breakdown")
    person_totals: list[PersonTotal] = Field(description="Total amount each person owes before tax")
    total_before_tax: float = Field(description="Total amount of the receipt before tax")
