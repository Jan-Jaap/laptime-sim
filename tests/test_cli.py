def test_cli_entrypoint_importable():
    from laptime_sim.cli import main

    assert callable(main)
