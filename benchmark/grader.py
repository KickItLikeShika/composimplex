"""Correctness graders for reasoning, coding and instruction following.

Math/GPQA grading is local; coding tasks use the official execution graders.
"""

from __future__ import annotations

import re
import signal
import json
import hashlib
from collections import defaultdict
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Optional

try:
    import sympy
    from sympy.parsing import sympy_parser
except Exception:  # pragma: no cover - optional benchmark dependency
    sympy = None
    sympy_parser = None

try:
    from pylatexenc import latex2text
except Exception:  # pragma: no cover - optional normalization enhancement
    latex2text = None


BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"
MATH_GRADING_TIMEOUT_SECONDS = 3.0
CODE_BENCHMARKS = ("LIVECODEBENCH_V6",)
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_GRADER_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
IFEVAL_GRADER_SHA256 = {
    "evaluation_lib": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "instructions": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instructions_registry": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instructions_util": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
}


@lru_cache(maxsize=1)
def load_ifeval_benchmark() -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("google/IFEval", "ifeval_input_data.jsonl",
                           repo_type="dataset", revision=IFEVAL_REVISION)
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def load_ifeval_grader():
    """Cache four unmodified, hash-verified Google modules; no pip package exists."""
    import importlib
    import sys
    import types
    from urllib.request import urlopen

    try:
        import absl.logging  # noqa: F401
        import immutabledict  # noqa: F401
        import langdetect
        import nltk
    except ImportError as exc:
        raise ImportError("IFEval requires `pip install -e '.[benchmark,ifeval]'`.") from exc
    from filelock import FileLock
    from huggingface_hub import cached_assets_path

    root = cached_assets_path("composimplex", namespace="ifeval",
                              subfolder=IFEVAL_GRADER_REVISION)
    with FileLock(str(root) + ".lock"):
        for name, expected_hash in IFEVAL_GRADER_SHA256.items():
            path = root / f"{name}.py"
            if path.exists():
                source = path.read_bytes()
            else:
                url = ("https://raw.githubusercontent.com/google-research/google-research/"
                       f"{IFEVAL_GRADER_REVISION}/instruction_following_eval/{name}.py")
                with urlopen(url, timeout=60) as response:
                    source = response.read()
            if hashlib.sha256(source).hexdigest() != expected_hash:
                raise RuntimeError(f"IFEval source checksum mismatch: {path}")
            if not path.exists():
                path.write_bytes(source)
        nltk_dir = root / "nltk_data"
        nltk.data.path.insert(0, str(nltk_dir))
        try:
            nltk.data.find("tokenizers/punkt_tab/english/")
        except LookupError:
            nltk.download("punkt_tab", download_dir=str(nltk_dir),
                          quiet=True, raise_on_error=True)

    # Google's imports require this namespace; keep the upstream files unchanged.
    for name in IFEVAL_GRADER_SHA256:
        loaded = sys.modules.get(f"instruction_following_eval.{name}")
        if loaded is not None and Path(loaded.__file__).resolve() != (root / f"{name}.py").resolve():
            raise RuntimeError("A different IFEval grader is already imported; use a fresh Python process.")
    package = types.ModuleType("instruction_following_eval")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    langdetect.DetectorFactory.seed = 0
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def grade_ifeval_records(records: list[dict]) -> None:
    grader = load_ifeval_grader()
    for record in records:
        inp = grader.InputExample(prompt=record["prompt"], **record["grading_input"])
        # Grade each sample separately: a shared prompt->response dict loses n>1.
        response = {inp.prompt: record["completion"]}
        strict = grader.test_instruction_following_strict(inp, response)
        loose = grader.test_instruction_following_loose(inp, response)
        record["correct"] = strict.follow_all_instructions
        record["answer"] = record["completion"]
        record["grading_result"] = {
            "strict": strict.follow_instruction_list,
            "loose": loose.follow_instruction_list,
            "strict_all": strict.follow_all_instructions,
            "loose_all": loose.follow_all_instructions,
        }
        record["grader"] = {
            "name": "google-research/instruction_following_eval",
            "revision": IFEVAL_GRADER_REVISION,
            "mode": "strict",
            "language_detection_seed": 0,
        }


@lru_cache(maxsize=1)
def load_code_benchmark(benchmark: str) -> list[dict]:
    if benchmark == "LIVECODEBENCH_V6":
        from huggingface_hub import hf_hub_download

        # Direct JSONL loading also works with datasets >= 4 (no dataset scripts).
        path = hf_hub_download(
            "livecodebench/code_generation_lite", "test6.jsonl",
            repo_type="dataset", revision=LCB_REVISION,
        )
        with open(path, encoding="utf-8") as handle:
            return sorted((json.loads(line) for line in handle if line.strip()),
                          key=lambda item: str(item["question_id"]))
    raise ValueError(f"Unsupported coding benchmark: {benchmark}")


def code_prompt(item: dict, benchmark: str) -> str:
    if benchmark not in CODE_BENCHMARKS:
        raise ValueError(f"Unsupported coding benchmark: {benchmark}")
    # Official LiveCodeBench GenericBase one-shot prompt, without chat roles.
    from lcb_runner.benchmarks import code_generation

    kind = "func" if item["starter_code"] else "stdin"
    examples = json.loads(Path(code_generation.__file__).parents[1].joinpath(
        f"prompts/few_shot_examples/generation/{kind}.json"
    ).read_text(encoding="utf-8"))

    def format_example(question, starter, answer):
        prompt = f"### Question\n{question}\n\n"
        if item["starter_code"]:
            prompt += f"### Starter Code\n{starter}\n\n"
        return prompt + "### Answer\n\n" + answer + ("\n\n" if answer else "")

    example = examples[0]
    return format_example(example["question"], example.get("sample_code", ""),
                          example["answer"]) + format_example(
        item["question_content"], item["starter_code"], ""
    )


def extract_code_solution(completion: str, problem: dict, benchmark: str) -> str:
    if benchmark not in CODE_BENCHMARKS:
        raise ValueError(f"Unsupported coding benchmark: {benchmark}")
    # Keep the raw response separately.
    completion = re.split(r"<\|turn>|<turn\|>|<eos>|<\|endoftext\|>", completion)[0]
    # Accept both raw code and fenced code from chat models.
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", completion, re.DOTALL | re.IGNORECASE)
    if blocks:
        completion = blocks[-1]
    return completion.split("###", 1)[0].strip()


def grade_code_records(records: list[dict], benchmark: str, config: dict) -> None:
    """Attach official execution results without changing our prefix pass@k."""
    if benchmark not in CODE_BENCHMARKS:
        raise ValueError(f"Unsupported coding benchmark: {benchmark}")
    if not records:
        return
    problems = {
        str(item["question_id"]): item
        for item in load_code_benchmark(benchmark)
    }
    workers = int(config.get("num_workers", 4))
    timeout = float(config.get("timeout", 6))
    if workers < 1 or timeout <= 0:
        raise ValueError("Evaluation workers and timeout must be positive.")
    distribution = metadata.distribution("livecodebench")
    grader_version = distribution.version
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        if commit:
            grader_version += ":" + commit
    from lcb_runner.evaluation import testing_util

    sources = sorted(Path(testing_util.__file__).parent.glob("*.py"))
    grader_version += ":sha256:" + hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    if not timeout.is_integer():
        raise ValueError("LiveCodeBench timeout must be an integer number of seconds.")
    timeout = int(timeout)
    for record in records:
        problem = problems[str(record["task_id"])]
        record["answer"] = extract_code_solution(record["completion"], problem, benchmark)
        record["grader"] = {"name": "livecodebench", "version": grader_version, "timeout": timeout}
        record["dataset_revision"] = LCB_REVISION

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation import codegen_metrics

    groups = defaultdict(list)
    for record in records:
        groups[str(record["task_id"])].append(record)
    groups = [sorted(group, key=lambda r: r["sample_index"])
              for _, group in sorted(groups.items())]
    if len({len(group) for group in groups}) > 1:
        raise ValueError("LiveCodeBench grading requires equal sample counts per problem.")
    samples = [CodeGenerationProblem(**problems[str(group[0]["task_id"])]).get_evaluation_sample()
               for group in groups]
    _, results, details = codegen_metrics(
        samples, [[r["answer"] for r in group] for group in groups],
        k_list=[1], num_process_evaluate=workers, timeout=timeout,
    )
    for i, group in enumerate(groups):
        for j, record in enumerate(group):
            tests = results[i][j]
            detail = json.loads(details[i][j])
            if detail.get("error_code") == -5:
                raise RuntimeError(f"LiveCodeBench test runner failed: {detail}")
            record["correct"] = bool(tests) and all(test > 0 for test in tests)
            record["grading_result"] = {"tests": tests, "metadata": detail}


def grade_outputs(records: list[dict], config: dict, outputs_path: Path,
                  regrade: bool = False) -> Path:
    if config["benchmark"] not in (*CODE_BENCHMARKS, "IFEVAL"):
        return outputs_path
    if not regrade and all(isinstance(r.get("correct"), bool) for r in records):
        return outputs_path
    if config["benchmark"] == "IFEVAL":
        grade_ifeval_records(records)
    else:
        grade_code_records(records, config["benchmark"], config.get("evaluation", {}))
    graded_path = outputs_path.with_name("graded_outputs.jsonl")
    with graded_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return graded_path


class _MathGradeTimeout(Exception):
    pass


def _math_grade_alarm_handler(signum, frame):
    del signum, frame
    raise _MathGradeTimeout()


def extract_all_boxed_inners(text: str) -> list[str]:
    if not text:
        return []
    results = []
    for match in re.finditer(r"\\(?:boxed|fbox)\{", text):
        start = match.end()
        depth = 1
        index = start
        while depth > 0 and index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            results.append(text[start : index - 1].strip())
    results.reverse()
    return results


def extract_last_math_expression(text: str) -> Optional[str]:
    if not text:
        return None
    dollar_matches = list(re.finditer(r"(?<!\\)\$([^$]+?)(?<!\\)\$", text))
    paren_matches = list(re.finditer(r"\\\((.*?)\\\)", text))
    candidates = [*dollar_matches, *paren_matches]
    if not candidates:
        return None
    return max(candidates, key=lambda match: match.start()).group(1)


def extract_math_answer(text: str) -> str:
    boxed = extract_all_boxed_inners(text)
    if boxed:
        return boxed[0]

    tag_matches = re.findall(
        r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL | re.IGNORECASE
    )
    if tag_matches:
        tagged = tag_matches[-1].strip()
        nested_boxed = extract_all_boxed_inners(tagged)
        return nested_boxed[0] if nested_boxed else tagged

    heuristic_matches = re.findall(
        r"(?:answer|equal to|is)\s+[:\s]?\s*([0-9a-zA-Z_^{}\\]+)(?:\.|$)",
        text or "",
        re.IGNORECASE,
    )
    if heuristic_matches:
        candidate = heuristic_matches[-1]
        if candidate.startswith("$") and candidate.endswith("$"):
            candidate = candidate[1:-1]
        return candidate

    last_math = extract_last_math_expression(text or "")
    if last_math:
        return last_math.split("=")[-1].strip() if "=" in last_math else last_math
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _sympy_parse(expression: str):
    if sympy_parser is None:
        raise RuntimeError("sympy is unavailable")
    return sympy_parser.parse_expr(
        expression.replace("^", "**"),
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expression: str) -> str:
    expression = expression.replace("\\tfrac", "\\frac")
    expression = expression.replace("\\dfrac", "\\frac")

    # Resolve innermost commands first so nested fractions retain grouping.
    previous = None
    while expression != previous:
        previous = expression
        expression = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            r"((\1)/(\2))",
            expression,
        )
        expression = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expression)

    if latex2text is not None and "\\" in expression:
        expression = latex2text.LatexNodes2Text().latex_to_text(expression)
    for source, target in {
        "√": "sqrt", "π": "pi", "∞": "inf", "∪": "U",
        "·": "*", "×": "*",
    }.items():
        expression = expression.replace(source, target)
    return expression.strip()


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _is_int(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _is_frac(expression: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expression))


def _strip_properly_formatted_commas(expression: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expression = pattern.sub(r"\1\3\4", expression)
        if next_expression == expression:
            return next_expression
        expression = next_expression


def _str_is_int(value: str) -> bool:
    try:
        number = float(_strip_properly_formatted_commas(value))
        return abs(number - int(round(number))) <= 1e-7
    except Exception:
        return False


def _str_to_int(value: str) -> int:
    return int(float(value.replace(",", "")))


def _inject_implicit_mixed_number(expression: str) -> str:
    return re.compile(r"([0-9]) +([0-9])").sub(r"\1+\2", expression)


def normalize_math_expression(expression: Optional[str]) -> Optional[str]:
    if expression is None:
        return None
    expression = str(expression).strip()
    text_match = re.search(r"^\\text\{(?P<text>.+?)\}$", expression)
    if text_match is not None:
        expression = text_match.group("text")

    expression = expression.replace("\\%", "%").replace("\\$", "$")
    expression = expression.replace("$", "").replace("%", "")
    expression = expression.replace(" or ", " , ").replace(" and ", " , ")
    expression = expression.replace("million", "*10^6")
    expression = expression.replace("billion", "*10^9")
    expression = expression.replace("trillion", "*10^12")

    for unit in (
        "degree", "cm", "centimeter", "meter", "mile", "second",
        "minute", "hour", "day", "week", "month", "year", "foot",
        "feet", "inch", "yard",
    ):
        expression = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expression)

    expression = re.sub(r"\^ *\\circ", "", expression)
    if len(expression) > 0 and expression[0] == "{" and expression[-1] == "}":
        expression = expression[1:-1]
    expression = re.sub(r",\\! *", "", expression)
    if _is_float(expression) and _is_int(float(expression)):
        expression = str(int(round(float(expression))))
    if "\\" in expression:
        try:
            expression = _parse_latex(expression)
        except Exception:
            pass

    expression = re.sub(r"- *", "-", expression)
    expression = _inject_implicit_mixed_number(expression)
    expression = expression.replace(" ", "").replace("{", "").replace("}", "").lower()
    if _str_is_int(expression):
        expression = str(_str_to_int(expression))
    return expression


def count_unknown_letters_in_expr(expression: str) -> int:
    expression = expression.replace("sqrt", "").replace("frac", "")
    return len({character for character in expression if character.isalpha()})


def should_allow_eval(expression: str) -> bool:
    if count_unknown_letters_in_expr(expression) > 2:
        return False
    if any(bad in expression for bad in BAD_SUBSTRINGS):
        return False
    return not any(re.search(pattern, expression) for pattern in BAD_REGEXES)


def _equation_zero_form(expression: str):
    if expression.count("=") != 1:
        return None
    left, right = expression.split("=", 1)
    return sympy.simplify(_sympy_parse(left) - _sympy_parse(right))


def are_equal_under_sympy(ground_truth: str, prediction: str) -> bool:
    if sympy is None:
        return False
    try:
        if "=" in ground_truth or "=" in prediction:
            ground_truth_zero = _equation_zero_form(ground_truth)
            prediction_zero = _equation_zero_form(prediction)
            if ground_truth_zero is None or prediction_zero is None:
                return False
            if sympy.simplify(ground_truth_zero - prediction_zero) == 0:
                return True
            ratio = sympy.simplify(ground_truth_zero / prediction_zero)
            return bool(ratio != 0 and not ratio.free_symbols)

        difference = f"({ground_truth})-({prediction})"
        if should_allow_eval(difference):
            return bool(sympy.simplify(_sympy_parse(difference)) == 0)
    except Exception:
        return False
    return False


def split_tuple(expression: str) -> list[str]:
    expression = _strip_properly_formatted_commas(expression)
    if not expression:
        return []
    if (
        len(expression) > 2
        and expression[0] in TUPLE_CHARS
        and expression[-1] in TUPLE_CHARS
        and all(character not in expression[1:-1] for character in TUPLE_CHARS)
    ):
        return [element.strip() for element in expression[1:-1].split(",")]
    return [expression]


def grade_math_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None:
        return False
    ground_truth_normalized = normalize_math_expression(ground_truth)
    given_normalized = normalize_math_expression(given_answer)
    if ground_truth_normalized is None or given_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if not given_normalized:
        return False

    ground_truth_elements = split_tuple(ground_truth_normalized)
    given_elements = split_tuple(given_normalized)
    if len(ground_truth_elements) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        return False
    if len(ground_truth_elements) != len(given_elements):
        return False

    for ground_truth_element, given_element in zip(
        ground_truth_elements, given_elements
    ):
        if _is_frac(ground_truth_element) and _is_frac(given_element):
            is_correct = ground_truth_element == given_element
        elif _str_is_int(ground_truth_element) != _str_is_int(given_element):
            is_correct = False
        else:
            is_correct = are_equal_under_sympy(ground_truth_element, given_element)
        if not is_correct:
            return False
    return True


def cheap_grade_math_answer(given_answer: str, ground_truth: str) -> bool:
    given_normalized = normalize_math_expression(given_answer)
    ground_truth_normalized = normalize_math_expression(ground_truth)
    return bool(
        given_normalized is not None
        and ground_truth_normalized is not None
        and given_normalized == ground_truth_normalized
    )


def safe_grade_math_answer(
    given_answer: str,
    ground_truth: str,
    timeout_seconds: float = MATH_GRADING_TIMEOUT_SECONDS,
) -> bool:
    if not hasattr(signal, "SIGALRM") or timeout_seconds <= 0:
        return grade_math_answer(given_answer, ground_truth)
    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _math_grade_alarm_handler)
    except ValueError:
        return grade_math_answer(given_answer, ground_truth)

    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return grade_math_answer(given_answer, ground_truth)
    except _MathGradeTimeout:
        return cheap_grade_math_answer(given_answer, ground_truth)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def parse_gpqa_answer(text: str) -> str:
    boxed = extract_math_answer(text).strip().upper()
    if boxed in {"A", "B", "C", "D"}:
        return boxed
    match = re.search(r"(?:answer|option)\s*[:=]?\s*([ABCD])\b", text, re.I)
    if match:
        return match.group(1).upper()
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    return matches[-1] if matches else "PARSE_ERROR"


def grade_completion(completion: str, reference: str, benchmark: str) -> bool:
    if benchmark == "MATH":
        return safe_grade_math_answer(
            extract_math_answer(completion),
            extract_math_answer(reference),
        )
    if benchmark == "GPQA":
        return parse_gpqa_answer(completion) == reference.strip().upper()
    raise ValueError(f"Unsupported benchmark: {benchmark}")
