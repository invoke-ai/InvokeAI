"""The Prompt Template node: where the prompt and the values of a template's own placeholders land."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from invokeai.app.invocations.fields import PromptTemplateValueField, StylePresetField
from invokeai.app.invocations.primitives import StringInvocation
from invokeai.app.invocations.prompt_template import (
    PromptTemplateInvocation,
    PromptTemplateValueInvocation,
    render_prompt_template,
)
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import CollectInvocation, Edge, EdgeConnection, Graph, InvalidEdgeError
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetData,
    PresetType,
    StylePresetWithoutId,
)
from invokeai.app.services.style_preset_records.style_preset_records_default import StylePresetRecordsStorage


def test_every_prompt_placeholder_takes_the_prompt_and_a_template_without_one_drops_it() -> None:
    assert render_prompt_template("{prompt}, again {prompt}", "a fox", {}) == "a fox, again a fox"
    assert render_prompt_template("oil painting", "a fox", {}) == "oil painting"


def test_a_placeholder_takes_its_value_whatever_its_case_and_nothing_when_it_has_none() -> None:
    template = "{Subject} and {subject}, {prompt}, {art_style} style"

    assert render_prompt_template(template, "at dusk", {"subject": "a fox"}) == "a fox and a fox, at dusk,  style"


def test_text_that_is_not_a_placeholder_is_left_as_written() -> None:
    template = "${season} {red|blue} {Prompt} {two words} {2d} {}"

    assert render_prompt_template(template, "a fox", {"season": "x", "red": "x", "prompt": "x"}) == template


def test_a_prompt_placeholder_after_a_dollar_sign_still_takes_the_prompt() -> None:
    assert render_prompt_template("${prompt}", "a fox", {}) == "$a fox"


def test_placeholders_inside_a_prompt_or_a_value_are_not_filled_again() -> None:
    values = {"subject": "{style} {prompt}", "style": "noir"}

    assert render_prompt_template("{subject} | {prompt}", "{style} \\1 $&", values) == (
        "{style} {prompt} | {style} \\1 $&"
    )


@pytest.mark.parametrize("name", ["{subject}", "two words", "2d", "", "prompt", "Prompt", "été", "subject\n"])
def test_a_value_node_with_a_name_that_is_not_a_placeholder_name_fails_with_the_rule(name: str) -> None:
    with pytest.raises(ValueError) as error:
        PromptTemplateValueInvocation(id="value", name=name, value="x").invoke(SimpleNamespace())  # type: ignore[arg-type]

    assert type(error.value) is ValueError
    assert str(error.value) == (
        f"'{name}' is not a prompt template placeholder name. Use letters A-Z, numbers and underscores, "
        "starting with a letter, without braces. 'prompt' is reserved."
    )


def test_a_value_given_directly_with_a_name_that_is_not_a_placeholder_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not a prompt template placeholder name"):
        PromptTemplateInvocation(
            id="template",
            style_preset=StylePresetField(style_preset_id="preset"),
            values=[{"name": "{style}", "value": "noir"}],  # type: ignore[list-item]
        )


def _context(database: Database, positive_prompt: str, negative_prompt: str) -> tuple[InvocationContext, str]:
    presets = StylePresetRecordsStorage(database)
    preset = presets.create(
        StylePresetWithoutId(
            name="Still",
            preset_data=PresetData(positive_prompt=positive_prompt, negative_prompt=negative_prompt),
            type=PresetType.User,
        ),
        user_id="system",
    )
    context = SimpleNamespace(_services=SimpleNamespace(style_preset_records=presets))

    return context, preset.id  # type: ignore[return-value]


def test_the_node_fills_both_sides_of_a_stored_template_from_a_collection_of_values(database: Database) -> None:
    context, preset_id = _context(database, "a still of {subject}. {prompt}. {style} style", "{prompt}, not {style}")
    subject = PromptTemplateValueInvocation(id="subject", name="subject", value="a fox").invoke(context).value
    style = PromptTemplateValueInvocation(id="style", name="Style", value="noir").invoke(context).value
    unused = PromptTemplateValueField(name="lighting", value="dusk")

    output = PromptTemplateInvocation(
        id="template",
        style_preset=StylePresetField(style_preset_id=preset_id),
        positive_prompt="at night",
        negative_prompt="blurry",
        values=[subject, style, unused],
    ).invoke(context)

    assert output.positive_prompt == "a still of a fox. at night. noir style"
    assert output.negative_prompt == "blurry, not noir"


def test_the_node_rejects_two_values_for_one_name(database: Database) -> None:
    context, preset_id = _context(database, "{prompt}, {style} style", "")
    template = PromptTemplateInvocation(
        id="template",
        style_preset=StylePresetField(style_preset_id=preset_id),
        values=[
            PromptTemplateValueField(name="Style", value="noir"),
            PromptTemplateValueField(name="style", value="pop"),
        ],
    )

    with pytest.raises(ValueError, match="more than one value for 'style'"):
        template.invoke(context)


def test_the_node_takes_one_value_or_none(database: Database) -> None:
    context, preset_id = _context(database, "{prompt}, {style} style", "")
    preset = StylePresetField(style_preset_id=preset_id)

    one = PromptTemplateInvocation(
        id="one",
        style_preset=preset,
        positive_prompt="a fox",
        values=PromptTemplateValueField(name="style", value="noir"),
    ).invoke(context)
    none = PromptTemplateInvocation(id="none", style_preset=preset, positive_prompt="a fox").invoke(context)

    assert one.positive_prompt == "a fox, noir style"
    assert none.positive_prompt == "a fox,  style"


def test_a_value_or_a_collection_of_values_connects_to_the_template_and_other_types_do_not() -> None:
    def edge(source: str, source_field: str, destination: str, destination_field: str = "values") -> Edge:
        return Edge(
            source=EdgeConnection(node_id=source, field=source_field),
            destination=EdgeConnection(node_id=destination, field=destination_field),
        )

    graph = Graph()
    for node in (
        PromptTemplateValueInvocation(id="value", name="style", value="noir"),
        CollectInvocation(id="collect"),
        StringInvocation(id="string", value="noir"),
        PromptTemplateInvocation(id="single", style_preset=StylePresetField(style_preset_id="preset")),
        PromptTemplateInvocation(id="collected", style_preset=StylePresetField(style_preset_id="preset")),
        PromptTemplateInvocation(id="mistyped", style_preset=StylePresetField(style_preset_id="preset")),
    ):
        graph.add_node(node)

    graph.add_edge(edge("value", "value", "single"))
    graph.add_edge(edge("value", "value", "collect", "item"))
    graph.add_edge(edge("collect", "collection", "collected"))
    graph.validate_self()

    with pytest.raises(InvalidEdgeError):
        graph.add_edge(edge("string", "value", "mistyped"))
