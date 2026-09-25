.PHONY: test validate quick paper replacement-quick replacement-paper llm-mock llm-h100 clean-quick

test:
	python -m unittest discover -s tests -v

validate:
	python -m tis validate

quick: test validate
	python -m tis run --config configs/public_quick.json --overwrite
	python -m tis verify-results --directory results/public_quick

paper: test validate
	python -m tis run --config configs/public_paper.json --overwrite
	python -m tis verify-results --directory results/public_paper

replacement-quick: test validate
	python -m tis run --config configs/replacement_quick.json --overwrite
	python -m tis verify-results --directory results/replacement_quick

replacement-paper: test validate
	python -m tis run --config configs/replacement_paper.json --overwrite
	python -m tis verify-results --directory results/replacement_paper

llm-mock: test
	python -m tis.llm_study.cli write-ci-mock --output build/llm_mock_kernels.json --questions 2
	python -m tis.llm_study.cli run --config configs/llm_study_mock_quick.json --kernels build/llm_mock_kernels.json --overwrite

llm-h100:
	bash scripts/run_llm_h100.sh

clean-quick:
	python -c "import shutil; shutil.rmtree('results/public_quick', ignore_errors=True)"
