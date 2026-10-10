
import json
from pathlib import Path
import requests

MODEL2_URL = "http://127.0.0.1:8001/recognize"
MODEL3_URL = "http://127.0.0.1:8000/translate"


def translate_with_model3(sequence):
    response = requests.post(
        MODEL3_URL,
        json={"sequence": sequence},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def recognize_with_model2(features):
    response = requests.post(
        MODEL2_URL,
        json={"features": features},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def main():
    print("\n===== SignBridge Integration =====")
    print("1. Test Model 3 using a sign sequence")
    print("2. Send existing features to Model 2, then Model 3")

    choice = input("Choose 1 or 2: ").strip()

    try:
        if choice == "1":
            sequence = input(
                "Enter recognized signs: "
            ).strip()

        elif choice == "2":
            filename = input(
                "Enter path to JSON features file: "
            ).strip()

            with open(Path(filename), "r", encoding="utf-8") as f:
                data = json.load(f)

            features = data["features"]
            result2 = recognize_with_model2(features)
            sequence = result2["sequence"]

            print("\nModel 2 output:", sequence)

        else:
            print("Invalid choice.")
            return

        if not sequence:
            print("No sign sequence received.")
            return

        result3 = translate_with_model3(sequence)

        print("\nRecognized signs:", sequence)
        print(
            "English translation:",
            result3.get("translation", result3)
        )
        print("\nIntegration test completed.")

    except FileNotFoundError:
        print("Feature file not found. Check the file path.")
    except (KeyError, ValueError) as error:
        print("Invalid input file:", error)
    except requests.RequestException as error:
        print("API error:", error)


if __name__ == "__main__":
    main()