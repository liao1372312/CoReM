"""Conservative source-level API dataflow sketches for train references.

Edges are static def-use candidates, not verified runtime dependencies.
Only API names and anonymous occurrence numbers are returned.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StaticDependencySketch:
    api_calls: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]


def _api_name(call: ast.Call) -> str | None:
    function = call.func
    if (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Attribute)
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "apis"
    ):
        return f"{function.value.attr}.{function.attr}"
    return None


def _assigned_names(target: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


class _Extractor:
    def __init__(self) -> None:
        self.sources: dict[str, set[int]] = {}
        self.calls: list[str] = []
        self.edges: set[tuple[int, int]] = set()

    def expression(self, node: ast.AST | None) -> set[int]:
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            return set(self.sources.get(node.id, set()))
        if isinstance(node, ast.NamedExpr):
            sources = self.expression(node.value)
            for name in _assigned_names(node.target):
                self.sources[name] = set(sources)
            return sources
        if isinstance(node, ast.Call):
            inputs: set[int] = set()
            for argument in node.args:
                inputs.update(self.expression(argument))
            for keyword in node.keywords:
                inputs.update(self.expression(keyword.value))
            name = _api_name(node)
            if name is None:
                inputs.update(self.expression(node.func))
                return inputs
            occurrence = len(self.calls) + 1
            self.calls.append(name)
            self.edges.update((producer, occurrence) for producer in inputs)
            return {occurrence}
        sources: set[int] = set()
        for child in ast.iter_child_nodes(node):
            sources.update(self.expression(child))
        return sources

    def statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.statements(statement.body)
        elif isinstance(statement, ast.Assign):
            sources = self.expression(statement.value)
            for target in statement.targets:
                for name in _assigned_names(target):
                    self.sources[name] = set(sources)
        elif isinstance(statement, ast.AnnAssign):
            sources = self.expression(statement.value)
            for name in _assigned_names(statement.target):
                self.sources[name] = set(sources)
        elif isinstance(statement, ast.AugAssign):
            sources = self.expression(statement.value)
            for name in _assigned_names(statement.target):
                self.sources[name] = self.sources.get(name, set()) | sources
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            sources = self.expression(statement.iter)
            for name in _assigned_names(statement.target):
                self.sources[name] = set(sources)
            self.statements(statement.body)
            self.statements(statement.orelse)
        elif isinstance(statement, ast.While):
            self.expression(statement.test)
            self.statements(statement.body)
            self.statements(statement.orelse)
        elif isinstance(statement, ast.If):
            self.expression(statement.test)
            original = {key: set(value) for key, value in self.sources.items()}
            self.statements(statement.body)
            after_body = self.sources
            self.sources = {key: set(value) for key, value in original.items()}
            self.statements(statement.orelse)
            for key, values in after_body.items():
                self.sources.setdefault(key, set()).update(values)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                sources = self.expression(item.context_expr)
                if item.optional_vars is not None:
                    for name in _assigned_names(item.optional_vars):
                        self.sources[name] = set(sources)
            self.statements(statement.body)
        elif isinstance(statement, ast.Try):
            self.statements(statement.body)
            for handler in statement.handlers:
                self.statements(handler.body)
            self.statements(statement.orelse)
            self.statements(statement.finalbody)
        elif isinstance(statement, ast.Match):
            self.expression(statement.subject)
            for case in statement.cases:
                self.statements(case.body)
        elif isinstance(statement, ast.Delete):
            for target in statement.targets:
                for name in _assigned_names(target):
                    self.sources.pop(name, None)
        else:
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    self.expression(child)


def extract_static_dependency_sketch(source: str) -> StaticDependencySketch:
    """Extract potential API-result-to-argument edges from parseable Python."""
    tree = ast.parse(source)
    extractor = _Extractor()
    extractor.statements(tree.body)
    return StaticDependencySketch(
        api_calls=tuple(extractor.calls),
        edges=tuple(sorted(extractor.edges)),
    )
