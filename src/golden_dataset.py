"""
Golden dataset generation for AI-agent conversation-quality evaluation.

Models the task Zendesk's posting describes: "dive into real-world
conversations to meticulously annotate and curate golden datasets... these
high-quality, benchmark datasets will be the source of truth for training
and testing our agents."

There is no real Zendesk (or any real company's) support-conversation data
available here, so this generates SYNTHETIC customer-support conversations
between a user and an AI support agent, across several realistic quality
tiers, and attaches a human-style annotation to each: a quality label
(good / borderline / bad) plus a short, specific reason — the same
structure a real annotator would produce, at a scale a real annotator
could actually check by hand (60 conversations, not thousands), so every
label in this "golden set" is inspectable and defensible rather than
mass-generated and rubber-stamped.
"""

import random
from dataclasses import dataclass, field

QUALITY_GOOD = "good"
QUALITY_BORDERLINE = "borderline"
QUALITY_BAD = "bad"


@dataclass
class Turn:
    speaker: str  # "user" or "agent"
    text: str


@dataclass
class Conversation:
    conversation_id: str
    turns: list = field(default_factory=list)
    # Ground-truth "human" annotation — the golden label.
    gold_label: str = None
    gold_reason: str = None

    def transcript(self):
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)


# --- Building blocks for synthetic conversations -----------------------

USER_ISSUES = [
    "my invoice from last month shows the wrong amount",
    "I can't log into my account, it says my password is invalid",
    "the export button on the reporting dashboard doesn't do anything",
    "I was charged twice for the same subscription",
    "how do I add a teammate to my workspace",
    "the API is returning a 500 error on the /tickets endpoint",
    "I need to cancel my subscription before the renewal date",
    "my data export is missing the last two weeks of tickets",
]

# A GOOD agent response: acknowledges the specific issue, gives a concrete
# next step or resolution, and doesn't invent facts not in evidence.
GOOD_RESPONSES = [
    "I can see the discrepancy on your account — it looks like a proration "
    "error from the plan change on the 14th. I've issued a correction; you "
    "should see it reflected within 1-2 business days, and here's a "
    "reference number for your records: REF-{ref}.",
    "That error usually means the cached session token expired. Could you "
    "try logging out completely and back in? If that doesn't resolve it, "
    "I can reset the token on our end — just confirm and I'll do that now.",
    "You're right, that's a bug — the export button isn't wired up for "
    "that report type yet. I've filed it as ticket {ref} for engineering "
    "and I'll follow up here once it's fixed. In the meantime, the CSV "
    "download under 'Legacy Reports' has the same data.",
]

# A BORDERLINE agent response: technically responsive, but vague, generic,
# or requires the user to do more work than necessary.
BORDERLINE_RESPONSES = [
    "Thanks for reaching out. Please try clearing your browser cache and "
    "cookies and let us know if the issue persists.",
    "I understand your concern. Our team is looking into similar reports "
    "and we'll update you when we know more.",
    "You can find information about that in our help center. Let me know "
    "if you have other questions!",
]

# A BAD agent response: ignores the actual question, gives wrong/irrelevant
# info, or is unhelpfully evasive.
BAD_RESPONSES = [
    "I'm not able to access billing information. Please contact your "
    "account manager.",  # deflects a question the agent should be able to answer
    "Have you tried restarting your computer?",  # irrelevant to an API 500 error
    "Great question! Unfortunately I don't have that information "
    "available right now.",
]


def _make_conversation(conv_id, issue, response_pool, label, reason):
    user_turn = Turn("user", f"Hi, I have a problem: {issue}.")
    ref = random.randint(10000, 99999)
    agent_text = random.choice(response_pool).format(ref=ref)
    agent_turn = Turn("agent", agent_text)
    convo = Conversation(conversation_id=conv_id, turns=[user_turn, agent_turn])
    convo.gold_label = label
    convo.gold_reason = reason
    return convo


def generate_golden_dataset(n_per_tier=20, seed=42):
    """
    Generates a golden dataset of synthetic support conversations with
    human-style ground-truth quality labels attached.

    n_per_tier=20 -> 60 conversations total (20 good, 20 borderline, 20 bad).
    Deliberately small: a golden set's value is in the labels being
    individually defensible, not in raw volume.
    """
    rng = random.Random(seed)
    conversations = []
    idx = 0

    for _ in range(n_per_tier):
        issue = rng.choice(USER_ISSUES)
        conversations.append(_make_conversation(
            f"conv-{idx:04d}", issue, GOOD_RESPONSES, QUALITY_GOOD,
            "Directly addresses the stated issue with a concrete resolution "
            "or clear, specific next step; no unsupported claims.",
        ))
        idx += 1

    for _ in range(n_per_tier):
        issue = rng.choice(USER_ISSUES)
        conversations.append(_make_conversation(
            f"conv-{idx:04d}", issue, BORDERLINE_RESPONSES, QUALITY_BORDERLINE,
            "Responsive in tone but generic — doesn't reference the "
            "specific issue and pushes work back to the user without a "
            "clear resolution path.",
        ))
        idx += 1

    for _ in range(n_per_tier):
        issue = rng.choice(USER_ISSUES)
        conversations.append(_make_conversation(
            f"conv-{idx:04d}", issue, BAD_RESPONSES, QUALITY_BAD,
            "Ignores or deflects the actual question; response is "
            "irrelevant or unhelpful given what the user asked.",
        ))
        idx += 1

    rng.shuffle(conversations)
    return conversations
