"""Тесты T5-T10 написаны как самостоятельные скрипты (выполняются при импорте и вызывают sys.exit),
поэтому pytest их собирать не должен: это ломает сбор тестов. Запуск: python -m eval.run_tests"""
collect_ignore = [
    "test_tools.py",
    "test_loader.py",
    "test_t10_tools.py",
    "test_agent.py",
    "test_snapshots.py",
]
