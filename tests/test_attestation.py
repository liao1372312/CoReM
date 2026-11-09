import unittest

from corem.attestation import (
    attest_workflow_hyperedge,
    partition_attested_hyperedges,
)


def workflow_rows() -> list[dict[str, object]]:
    return [
        {
            "memory_id": "step1",
            "role": "workflow_step",
            "step_index": 1,
            "operation_ids": ["lookup"],
            "plan": {"capture": {"name": "item_id"}},
        },
        {
            "memory_id": "step2",
            "role": "workflow_step",
            "step_index": 2,
            "operation_ids": ["lookup"],
            "plan": {
                "path": "/items/{item_id}",
                "capture": {"name": "item_name"},
            },
        },
        {
            "memory_id": "step3",
            "role": "workflow_step",
            "step_index": 3,
            "operation_ids": ["lookup"],
            "plan": {"path": "/search?name={item_name}"},
        },
        {
            "memory_id": "verify",
            "role": "verification",
            "step_index": 4,
            "operation_ids": ["lookup"],
            "verification_contract": {
                "handoffs": [
                    {"name": "item_id", "target_step": 2},
                    {"name": "item_name", "target_step": 3},
                ]
            },
        },
    ]


class WorkflowAttestationTests(unittest.TestCase):
    def test_accepts_grounded_runtime_handoffs(self) -> None:
        rows = workflow_rows()
        edge = {"edge_id": "valid", "members": [row["memory_id"] for row in rows]}
        accepted, reason = attest_workflow_hyperedge(
            {str(row["memory_id"]): row for row in rows}, edge
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "attested")

    def test_rejects_rewritten_member_that_breaks_binding(self) -> None:
        rows = workflow_rows()
        rows.append(
            {
                "memory_id": "wrong-step2",
                "role": "workflow_step",
                "step_index": 2,
                "operation_ids": ["alternative:lookup"],
                "plan": {"path": "/items/{other_id}"},
            }
        )
        edge = {
            "edge_id": "rewritten",
            "members": ["step1", "wrong-step2", "step3", "verify"],
        }
        accepted, reason = attest_workflow_hyperedge(
            {str(row["memory_id"]): row for row in rows}, edge
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "operation_identity_mismatch")

    def test_rejects_same_named_binding_from_another_operation(self) -> None:
        rows = workflow_rows()
        rows.append(
            {
                "memory_id": "other-step2",
                "role": "workflow_step",
                "step_index": 2,
                "operation_ids": ["alternative:lookup"],
                "plan": {
                    "path": "/other/{item_id}",
                    "capture": {"name": "item_name"},
                },
            }
        )
        edge = {
            "edge_id": "cross-operation",
            "members": ["step1", "other-step2", "step3", "verify"],
        }
        accepted, reason = attest_workflow_hyperedge(
            {str(row["memory_id"]): row for row in rows}, edge
        )
        self.assertFalse(accepted)
        self.assertEqual(reason, "operation_identity_mismatch")

    def test_partition_records_rejection_reason(self) -> None:
        rows = workflow_rows()
        valid = {"edge_id": "valid", "members": [row["memory_id"] for row in rows]}
        invalid = {"edge_id": "invalid", "members": ["step1", "step2"]}
        accepted, rejected = partition_attested_hyperedges(rows, [valid, invalid])
        self.assertEqual([row["edge_id"] for row in accepted], ["valid"])
        self.assertEqual(
            rejected,
            [{"edge_id": "invalid", "reason": "verification_contract_count"}],
        )


if __name__ == "__main__":
    unittest.main()
