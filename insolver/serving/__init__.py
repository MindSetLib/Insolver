from ..utils import check_dependency

deps = ["flask", "fastapi", "uvicorn", "pydantic", "gunicorn", "django", "rest_framework", "sympy", "jinja2"]

for dep in deps:
    check_dependency(package_name=dep, extra_name='serving')
