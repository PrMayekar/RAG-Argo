


# test_llm.py
from query import ask

TEST_CASES = [
    {
        "question": "What platform type is float 5907176?",
        "must_contain": ["ARVOR"],
        "description": "Float platform type lookup"
    },
    {
        "question": "What was the surface temperature recorded by float 5907176?",
        "must_contain": ["2.37", "°C"],
        "description": "Surface temperature value"
    },
    {
        "question": "What is the WMO ID of the SOLO_II platform float?",
        "must_contain": ["2903843"],
        "description": "WMO ID lookup by platform"
    },
    {
        "question": "What date did float 1901771 record data?",
        "must_contain": ["2026"],
        "description": "Date lookup"
    },
    {
        "question": "What is the deepest pressure recorded?",
        "must_contain": ["dbar"],
        "description": "Pressure unit present in answer"
    },
]

if __name__ == "__main__":
    print("\nLLM ANSWER QUALITY TESTS\n")
    passed = failed = 0

    for tc in TEST_CASES:
        print(f"Testing: {tc['description']}")
        print(f"  Q: {tc['question']}")
        try:
            answer = ask(tc["question"])
            print(f"  A: {answer[:120]}...")

            all_found = all(kw.lower() in answer.lower()
                           for kw in tc["must_contain"])

            if all_found:
                print(f"  PASS: Found keywords {tc['must_contain']}")
                passed += 1
            else:
                missing = [kw for kw in tc["must_contain"]
                          if kw.lower() not in answer.lower()]
                print(f"  FAIL: Missing keywords {missing}")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1
        print()

    print(f"{'='*35}")
    print(f"Results: {passed}/{len(TEST_CASES)} passed")
    accuracy = (passed / len(TEST_CASES)) * 100
    print(f"Accuracy: {accuracy:.0f}%")



