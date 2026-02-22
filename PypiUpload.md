## Pip upload  <!-- omit from toc -->

```bash
python3 -m pip install --upgrade pip build twine
rm -rf dist build *.egg-info
python3 -m build
python3 -m twine check dist/*
python3 -m pip install dist/*.whl
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-pypi-token> e.g. from password manager
python -m twine upload dist/*
```