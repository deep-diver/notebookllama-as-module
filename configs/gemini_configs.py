from google.ai.generativelanguage_v1beta.types import content

write_script_config = {
    "model_name": "gemini-1.5-flash-002",
    "generation_config": {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_schema": content.Schema(
            type=content.Type.OBJECT,
            required=["conversations"],
            properties={
                "conversations": content.Schema(
                    type=content.Type.ARRAY,
                    items=content.Schema(
                        type=content.Type.OBJECT,
                        required=["Alex", "Jamie"],
                        properties={
                            "Alex": content.Schema(
                                type=content.Type.STRING,
                            ),
                            "Jamie": content.Schema(
                                type=content.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
        "response_mime_type": "application/json",
    }
}
