def inspect_autobuild(autobuild_path, event):


    print("\tProcessing autobuilding results...")
    autobuilding_log_path = out_dir_path / "pandda_autobuild_log.txt"
    write_autobuild_log(formatted_command, stdout, stderr, autobuilding_log_path)

    try:
        result = AutobuildingResultRhofit.from_output(event,
                                                      stdout,
                                                      stderr,
                                                      )
    except:
        result = AutobuildingResultRhofit.null_result(event)

    return result
