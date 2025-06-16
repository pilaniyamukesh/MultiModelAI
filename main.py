from client import Client


def main():
    # Change these values to test different providers
    provider_name = "ollama"  # or "openai"
    model_name = "qwen2.5-coder:3b"
    api_key = None  # or provide your key if needed

    try:
        client = Client(provider=provider_name, model_name=model_name, api_key=api_key)
        #response = client.chat(["why the color of sky is blue?"])
        #print(f"Response from {provider_name}: {response}")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_price",
                "description": "Get the price of a product",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product": {
                            "type": "string",
                            "description": "The product's name.",
                        },
                        "price": {
                            "type": "int",
                            "description": "price of a product"
                        },
                    },
                },
            },
        }]

        #messages = [
        #    {"role": "user", "content": "What is the price of an iPhone 16?"}
        #]
        messages = [
            {
                "role": "user",
                "content": "why is the sky blue?"
            },
            {
                "role": "assistant",
                "content": "due to rayleigh scattering."
            },
            {
                "role": "user",
                "content": "how is that different than mie scattering?"
            }
        ]

        response = client.chat(
            message=messages,
            temperature= 1.0,
            #tools=tools
        )
        print(response)
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

