import subprocess


def event_map_to_mtz(event_map_path,
                     output_path,
                     resolution,
                     col_f="FWT",
                     col_ph="PHWT",
                     gemmi_path = "/dls/science/groups/i04-1/conor_dev/gemmi/gemmi",
                     ):
    command = "module load gcc/4.9.3; source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_clipper_no_mkl; {gemmi_path} map2sf {event_map_path} {output_path} {col_f} {col_ph} --dmin={resolution}"
    formatted_command = command.format(gemmi_path=gemmi_path,
                                       event_map_path=event_map_path,
                                       output_path=output_path,
                                       col_f=col_f,
                                       col_ph=col_ph,
                                       resolution=resolution,
                                       )
    # print(formatted_command)

    p = subprocess.Popen(formatted_command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         )

    stdout, stderr = p.communicate()

    return formatted_command, stdout, stderr