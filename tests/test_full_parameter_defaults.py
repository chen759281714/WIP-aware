"""Configuration-only checks: no optimizer construction, evaluation or decode."""

import ast
import inspect
import unittest
from pathlib import Path

from src.algorithms import wip_graph_dual_population as full
from src.algorithms.wip_graph_dual_population_wo_ac import WIPGraphDualPopulationWithoutAC
from src.algorithms.wip_graph_dual_population_wo_hlns import WIPGraphDualPopulationWithoutHLNS
from src.algorithms.wip_graph_dual_population_wo_mgs import WIPGraphDualPopulationWithoutMGS


class FullParameterDefaultsTests(unittest.TestCase):
    expected = dict(N=10, N_A=20, p_mut=0.35, rho=0.95, T_coop=3)

    @staticmethod
    def resolved_constructor_parameters(**overrides):
        signature = inspect.signature(full.WIPGraphDualPopulation.__init__)
        bound = signature.bind(None, {}, {}, variant="HLNS_FULL", **overrides)
        bound.apply_defaults()
        tree = ast.parse(inspect.getsource(full))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "WIPGraphDualPopulation")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        # Execute only the parameter-resolution guard, never the constructor.
        guard = next(n for n in init.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "N_A is None")
        if any(isinstance(n, ast.Call) for n in ast.walk(guard)):
            raise AssertionError("Archive default resolution must not call runtime code")
        exec(compile(ast.Module(body=[guard], type_ignores=[]), "<default-resolution>", "exec"),
             {}, bound.arguments)
        return {key: bound.arguments[key] for key in FullParameterDefaultsTests.expected}

    def test_default_parameters(self):
        self.assertEqual(full.get_default_full_parameters(), self.expected)

    def test_shared_defaults_are_read_only(self):
        with self.assertRaises(TypeError):
            full.DEFAULT_FULL_PARAMETERS["N"] = 99

    def test_returned_parameters_are_independent(self):
        values = full.get_default_full_parameters()
        values["N"] = 99
        self.assertEqual(full.get_default_full_parameters(), self.expected)

    def test_population_override_resolves_archive(self):
        for n in (5, 10, 20, 30, 50):
            with self.subTest(N=n):
                self.assertEqual(full.get_default_full_parameters(N=n)["N_A"], 2*n)

    def test_explicit_archive_is_independent(self):
        self.assertEqual(full.get_default_full_parameters(N=20, N_A=77)["N_A"], 77)
        self.assertEqual(full.get_default_full_parameters(N_A=33)["N_A"], 33)

    def test_explicit_search_parameters(self):
        values = full.get_default_full_parameters(p_mut=0.1, rho=0.8, T_coop=10)
        self.assertEqual((values["p_mut"], values["rho"], values["T_coop"]), (0.1, 0.8, 10))

    def test_none_archive_resolves_after_population(self):
        self.assertEqual(full.get_default_full_parameters(N=20, N_A=None)["N_A"], 40)

    def test_unknown_configuration_key_is_rejected(self):
        with self.assertRaises(TypeError):
            full.get_default_full_parameters(p_mu=0.1)

    def test_constructor_defaults_match_shared_entry(self):
        self.assertEqual(self.resolved_constructor_parameters(), self.expected)

    def test_constructor_population_override(self):
        self.assertEqual(self.resolved_constructor_parameters(N=20)["N_A"], 40)

    def test_constructor_explicit_parameters_win(self):
        explicit = dict(N=50, N_A=12, p_mut=0.1, rho=0.8, T_coop=10)
        self.assertEqual(self.resolved_constructor_parameters(**explicit), explicit)

    def test_variants_retain_inheritance_and_switches(self):
        cases = (
            (full.WIPGraphDualPopulation, ("HLNS_FULL", "MGS", "HLNS", True)),
            (WIPGraphDualPopulationWithoutMGS, ("HLNS_WO_MGS", "BasicMutation", "HLNS", True)),
            (WIPGraphDualPopulationWithoutHLNS, ("HLNS_WO_HLNS", "MGS", "BasicMutation", True)),
            (WIPGraphDualPopulationWithoutAC, ("HLNS_WO_AC", "MGS", "HLNS", False)),
        )
        for cls, expected in cases:
            with self.subTest(variant=expected[0]):
                self.assertTrue(issubclass(cls, full.WIPGraphDualPopulation))
                instance = object.__new__(cls)
                self.assertEqual((instance.algorithm_variant, instance.pm_specialized_kind,
                                  instance.ps_specialized_kind, instance.pm_archive_guidance_enabled), expected)
                self.assertFalse(instance.ps_archive_guidance_enabled)

    def test_taguchi_keeps_all_explicit_parameters(self):
        path = Path(full.__file__).resolve().parents[2]/"experiments/run_taguchi_tuning.py"
        tree = ast.parse(path.read_text())
        run_task = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_task")
        assignment = next(n for n in run_task.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "params" for t in n.targets))
        keys = next(n for n in ast.walk(assignment) if isinstance(n, ast.Tuple))
        self.assertEqual(set(ast.literal_eval(keys)),
                         {"N", "N_A", "p_mut", "rho", "T_coop", "FE_max", "seed"})
        call = next(n for n in ast.walk(run_task) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == "WIPGraphDualPopulation")
        self.assertTrue(any(k.arg is None and isinstance(k.value, ast.Name)
                            and k.value.id == "params" for k in call.keywords))
        self.assertTrue(any(k.arg == "variant" and isinstance(k.value, ast.Constant)
                            and k.value.value == "HLNS_FULL" for k in call.keywords))


if __name__ == "__main__":
    unittest.main()
