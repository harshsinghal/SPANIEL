PYTHON ?= python3

.PHONY: test verify-artifacts build-eval build-v2 build-v2-without-synth build-context

test:
	git ls-files -z "*.py" | xargs -0 $(PYTHON) -m py_compile
	$(PYTHON) -m unittest discover -v

verify-artifacts:
	$(PYTHON) verify_artifacts.py

build-eval:
	$(PYTHON) build_sft_dataset.py --outdir sft_data

build-v2:
	$(PYTHON) convert_ai4privacy.py --out pii_xml/ai4privacy_en.jsonl
	$(PYTHON) build_sft_dataset_v2.py --outdir sft_data_v2

build-v2-without-synth:
	$(PYTHON) convert_ai4privacy.py --out pii_xml/ai4privacy_en.jsonl
	$(PYTHON) build_sft_dataset_v2.py --skip-synth --outdir sft_data_v2

build-context:
	$(PYTHON) -m sensitive_context.build --out context-data
