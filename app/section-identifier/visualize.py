import sweetviz as sv

def make_report(dataset, unq_id):
    my_report = sv.analyze(dataset) # ,target_feat="label"
    my_report.show_html(f"Dataset {unq_id} report.html")