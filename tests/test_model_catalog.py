from __future__ import annotations

import pytest

from ness_cli.model_catalog import parse_catalog


def test_parse_catalog_filters_and_preserves_literal_reasoning() -> None:
    openrouter = {
        "data": [
            {
                "id": "z-ai/glm-5.2",
                "name": "GLM 5.2",
                "context_length": 1_048_576,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["tools", "reasoning_effort"],
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000003",
                    "input_cache_read": "0.0000002",
                },
            },
            {
                "id": "image/generator",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                "supported_parameters": ["tools"],
            },
            {
                "id": "text/no-tools",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": [],
            },
        ]
    }
    models_dev = {
        "zhipuai": {
            "models": {
                "glm-5.2": {
                    "reasoning_options": [
                        {"type": "effort", "values": ["high", "max"]}
                    ]
                }
            }
        }
    }

    records = parse_catalog(openrouter, models_dev)

    assert [record.id for record in records] == ["z-ai/glm-5.2"]
    assert records[0].reasoning_efforts == ("high", "max")
    assert records[0].context_length == 1_048_576
    assert records[0].input_price == 1.0
    assert records[0].cache_read_ratio == pytest.approx(0.2)


def test_parse_catalog_marks_vision_and_anthropic_messages() -> None:
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-5",
                "name": "Claude Sonnet 5",
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["tools"],
            }
        ]
    }

    record = parse_catalog(payload)[0]

    assert record.supports_vision is True
    assert record.supports_anthropic_messages is True
