import os

benchmark_suites = set([_ for _ in os.listdir('Benchmarks')])
data_suites = set([_ for _ in os.listdir('Data')])
suite_matches = set(benchmark_suites).intersection(set(data_suites))
for suite in suite_matches:
    benchmarks = set([_ for _ in os.listdir(f'Benchmarks/{suite}')])
    datas = set([_ for _ in os.listdir(f'Data/{suite}')])
    matches = benchmarks.intersection(datas)
    for match in matches:
        try:
            os.symlink(f'../../../Data/{suite}/{match}/', f'Benchmarks/{suite}/{match}/Data')
            print(f"Successfully linked {suite} - {match}")
        except FileExistsError:
            if os.readlink(f'Benchmarks/{suite}/{match}/Data') == f'../../../Data/{suite}/{match}/':
                print(f"Symlink for {suite} - {match} exists -- OK")
            else:
                print(f"!! WARNING: File 'Data' already exists in Benchmarks/{suite}/{match}. NO LINK CREATED !!")
    for unmatched in benchmarks.difference(matches):
        print(f"Unmatched Benchmark: {suite}/{unmatched}")
    for unmatched in datas.difference(matches):
        print(f"Unmatched Data: {suite}/{unmatched}")
for unmatched in benchmark_suites.difference(suite_matches):
    print(f"Unmatched Benchmark Suite: {unmatched}")
for unmatched in data_suites.difference(suite_matches):
    print(f"Unmatched Data Suite: {unmatched}")
