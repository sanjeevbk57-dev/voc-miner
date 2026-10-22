"""Shared theme list, used by classify_reviews.py and the eval.

This is a PM decision: it was drafted from what clustering discovered,
then fixed so results are stable and hand-labels can be compared against it.
"""
THEMES = {
    "Missing or wrong items": "items missing from the order, or a different item sent",
    "Order not delivered or cancelled": "order never arrived, or was cancelled after payment",
    "Expired, rotten or poor quality": "expired, spoiled, rotten, fake or low-quality products",
    "Damaged or leaking packaging": "torn bags, leaks, broken seals, melted or warm items",
    "Late delivery": "slower than the promised time, wrong ETAs",
    "Fees and charges": "delivery, handling, surge, platform or small-order fees",
    "Prices and offers": "high prices, price changes, offers or cashback not working",
    "Payment problems": "failed payments, double debits, cash on delivery unavailable",
    "App, account or location": "app crashes, login, account blocked, area not serviceable, referral",
    "Delivery partner behaviour": "rude or unresponsive delivery partners",
    "Support and refunds": "ONLY when the main complaint is support itself or a refund, with no clearer root problem above",
    "Generic complaint": "vague 'worst app' reviews with no specific problem",
    "Off-topic": "restaurant food delivery, not groceries or quick commerce",
}
