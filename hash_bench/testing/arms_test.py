# Copyright 2026 The hash-bench Authors. SPDX-License-Identifier: Apache-2.0
"""The arm vocabulary: what each arm asks the compiler for, and what a compiled
module has to look like for the harness to say the arm ran."""

from absl.testing import absltest

from hash_bench import arms

# Trimmed from a real compiled module; the fields the classifier reads are the
# fusion kind and the called computation, and everything else is noise it must
# tolerate.
_DEDICATED = (
    "%poseidon2 = koalabear_mont[4096]{0} fusion(%bitcast, %wrapped), "
    'kind=kCustom, calls=%poseidon2_fusion, backend_config={"kind":"__custom_fusion"}'
)
_STATIC_WHILE = (
    "ROOT %fusion = koalabear_mont[256,16]{1,0} fusion(%state.1), "
    "kind=kCustom, calls=%static_while_fusion, "
    'backend_config={"kind":"__custom_fusion"}'
)
_STATIC_WHILE_SUFFIXED = _STATIC_WHILE.replace(
    "calls=%static_while_fusion", "calls=%static_while_fusion.2"
)
_LOOP = "%fused = f32[8]{0} fusion(%a), kind=kLoop, calls=%fused_computation"


class ClassifyTest(absltest.TestCase):
    def test_no_custom_fusion_is_the_decomposition(self) -> None:
        lowering, callees = arms.observe(_LOOP + "\n" + _LOOP)
        self.assertEqual(lowering, arms.Lowering.DECOMPOSED)
        self.assertEqual(lowering.arm_name, "declined")
        self.assertEmpty(callees)

    def test_a_named_kernel_is_the_dedicated_arm(self) -> None:
        lowering, callees = arms.observe(_DEDICATED + "\n" + _LOOP)
        self.assertEqual(lowering, arms.Lowering.DEDICATED)
        self.assertEqual(lowering.arm_name, "routed")
        self.assertEqual(callees, ("poseidon2_fusion",))

    def test_the_loop_conversion_is_the_generic_arm(self) -> None:
        lowering, _ = arms.observe(_STATIC_WHILE)
        self.assertEqual(lowering, arms.Lowering.STATIC_WHILE)
        self.assertEqual(lowering.arm_name, "generic")

    def test_a_disambiguating_suffix_is_still_the_loop_conversion(self) -> None:
        # A module with several converted loops numbers them; reading the suffix
        # as part of the name would report every such module as dedicated.
        lowering, _ = arms.observe(_STATIC_WHILE_SUFFIXED)
        self.assertEqual(lowering, arms.Lowering.STATIC_WHILE)

    def test_a_dedicated_kernel_beside_a_converted_loop_is_mixed(self) -> None:
        lowering, callees = arms.observe(_DEDICATED + "\n" + _STATIC_WHILE)
        self.assertEqual(lowering, arms.Lowering.MIXED)
        self.assertLen(callees, 2)


class FlagsTest(absltest.TestCase):
    def test_routed_disables_nothing(self) -> None:
        self.assertEqual(arms.Arm.ROUTED.xla_flags, ())

    def test_generic_disables_every_recognizer_and_keeps_the_conversion(self) -> None:
        (flag,) = arms.Arm.GENERIC.xla_flags
        disabled = flag.split("=", 1)[1].split(",")
        self.assertCountEqual(disabled, arms.RECOGNIZER_PASSES)

    def test_declined_also_disables_the_loop_conversion(self) -> None:
        (flag,) = arms.Arm.DECLINED.xla_flags
        disabled = flag.split("=", 1)[1].split(",")
        self.assertCountEqual(
            disabled, (*arms.RECOGNIZER_PASSES, arms.LOOP_CONVERSION_PASS)
        )

    def test_every_arm_names_a_lowering(self) -> None:
        # The two enums are separate so a row can carry a request and an
        # observation; they still have to speak the same words.
        lowerings = {l.arm_name for l in arms.Lowering}
        for arm in arms.Arm:
            self.assertIn(arm.value, lowerings)


if __name__ == "__main__":
    absltest.main()
