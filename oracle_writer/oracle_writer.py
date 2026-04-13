"""
oracle_writer.py

Writes AI agent scores to CampaignV2 contracts on-chain.
Called by the synthesis agent after all signals are aggregated.

Usage:
    writer = OracleWriter()
    tx_hash = writer.submit_score(campaign_address, milestone_index, score)
"""

import os
import json
from pathlib import Path
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()


# Minimal ABI — only the functions the oracle needs
CAMPAIGN_ABI = [
    {
        "inputs": [
            {"name": "milestoneIndex", "type": "uint256"},
            {"name": "score",          "type": "uint256"}
        ],
        "name": "submitAgentScore",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "currentMilestoneIndex",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "index", "type": "uint256"}],
        "name": "getMilestone",
        "outputs": [
            {"name": "desc",               "type": "string"},
            {"name": "fundingBps",         "type": "uint256"},
            {"name": "state",              "type": "uint8"},
            {"name": "agentScore",         "type": "uint256"},
            {"name": "agentScoreSubmitted","type": "bool"},
            {"name": "commitDeadline",     "type": "uint256"},
            {"name": "revealDeadline",     "type": "uint256"},
            {"name": "ethAllocation",      "type": "uint256"},
            {"name": "fundsReleased",      "type": "bool"},
            {"name": "participantCount",   "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "campaignState",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class OracleWriter:
    """
    Connects to Tenderly testnet and submits AI agent scores
    to CampaignV2 contracts as the authorized oracle address.
    """

    # MilestoneState enum values matching the contract
    MILESTONE_STATE = {
        0: "Pending",
        1: "VotingOpen",
        2: "RevealOpen",
        3: "Passed",
        4: "Failed",
        5: "Inconclusive"
    }

    # CampaignState enum values
    CAMPAIGN_STATE = {
        0: "Active",
        1: "Funded",
        2: "Completed",
        3: "Cancelled"
    }

    def __init__(self):
        rpc_url     = os.getenv("RPC_URL")
        private_key = os.getenv("ORACLE_PRIVATE_KEY")

        if not rpc_url:
            raise ValueError("RPC_URL not set in .env")
        if not private_key:
            raise ValueError("ORACLE_PRIVATE_KEY not set in .env")

        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to RPC: {rpc_url}")

        self.account = self.w3.eth.account.from_key(private_key)
        print(f"Oracle address: {self.account.address}")

    def get_campaign(self, campaign_address: str):
        """Return a contract instance for a deployed CampaignV2."""
        checksum_addr = Web3.to_checksum_address(campaign_address)
        return self.w3.eth.contract(
            address=checksum_addr,
            abi=CAMPAIGN_ABI
        )

    def get_milestone_info(self, campaign_address: str, milestone_index: int) -> dict:
        """
        Fetch current milestone state from chain.
        Returns a dict with all milestone fields.
        """
        campaign = self.get_campaign(campaign_address)
        result   = campaign.functions.getMilestone(milestone_index).call()

        return {
            "description":        result[0],
            "funding_bps":        result[1],
            "state":              self.MILESTONE_STATE.get(result[2], "Unknown"),
            "state_raw":          result[2],
            "agent_score":        result[3],
            "agent_score_submitted": result[4],
            "commit_deadline":    result[5],
            "reveal_deadline":    result[6],
            "eth_allocation":     result[7],
            "funds_released":     result[8],
            "participant_count":  result[9],
        }

    def get_current_milestone_index(self, campaign_address: str) -> int:
        """Return the active milestone index for a funded campaign."""
        campaign = self.get_campaign(campaign_address)
        return campaign.functions.currentMilestoneIndex().call()

    def can_submit_score(self, campaign_address: str, milestone_index: int) -> tuple[bool, str]:
        """
        Check whether submitting a score is currently valid.
        Returns (can_submit: bool, reason: str).
        """
        try:
            info = self.get_milestone_info(campaign_address, milestone_index)
        except Exception as e:
            return False, f"Could not fetch milestone: {e}"

        if info["agent_score_submitted"]:
            return False, "Agent score already submitted for this milestone"

        if info["state_raw"] != 0:  # 0 = Pending
            return False, f"Milestone is in {info['state']} state, must be Pending"

        return True, "OK"

    def submit_score(
        self,
        campaign_address: str,
        milestone_index: int,
        score: int
    ) -> str:
        """
        Submit an AI agent verification score to a CampaignV2 contract.

        Args:
            campaign_address: Deployed CampaignV2 address
            milestone_index:  Which milestone to score (usually currentMilestoneIndex)
            score:            0–10000. 10000 = agent certain milestone achieved.

        Returns:
            Transaction hash as hex string.

        Raises:
            ValueError: If score out of range or submission not currently valid.
            Exception:  If transaction fails.
        """
        # Validate score range
        if not (0 <= score <= 10_000):
            raise ValueError(f"Score must be 0–10000, got {score}")

        # Pre-flight check
        can_submit, reason = self.can_submit_score(campaign_address, milestone_index)
        if not can_submit:
            raise ValueError(f"Cannot submit score: {reason}")

        campaign = self.get_campaign(campaign_address)
        checksum_addr = Web3.to_checksum_address(campaign_address)

        # Build transaction
        nonce = self.w3.eth.get_transaction_count(self.account.address)

        txn = campaign.functions.submitAgentScore(
            milestone_index,
            score
        ).build_transaction({
            "from":     self.account.address,
            "nonce":    nonce,
            "gas":      100_000,
            "gasPrice": self.w3.eth.gas_price,
        })

        # Sign and send
        signed = self.w3.eth.account.sign_transaction(
            txn, self.account.key
        )
        tx_hash = self.w3.eth.send_raw_transaction(
            signed.raw_transaction
        )

        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=120
        )

        if receipt.status != 1:
            raise Exception(
                f"Transaction failed. Hash: {tx_hash.hex()}"
            )

        print(f"Score submitted successfully.")
        print(f"  Campaign:  {checksum_addr}")
        print(f"  Milestone: {milestone_index}")
        print(f"  Score:     {score} / 10000")
        print(f"  TX hash:   {tx_hash.hex()}")

        return tx_hash.hex()


# ── Quick test / manual invocation ─────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python oracle_writer.py <campaign_address> <milestone_index> <score>")
        print("Example: python oracle_writer.py 0xABC...123 0 7500")
        sys.exit(1)

    campaign_addr   = sys.argv[1]
    milestone_idx   = int(sys.argv[2])
    agent_score     = int(sys.argv[3])

    writer = OracleWriter()

    # Show current state before submitting
    info = writer.get_milestone_info(campaign_addr, milestone_idx)
    print(f"\nMilestone {milestone_idx} current state: {info['state']}")
    print(f"Description: {info['description']}")
    print(f"Score already submitted: {info['agent_score_submitted']}")

    # Submit
    tx = writer.submit_score(campaign_addr, milestone_idx, agent_score)
    print(f"\nDone. TX: {tx}")