# WIP: Basic shape works, next I'll have to add checkpoint loading and preprocessing (if uncached) and maybe a loading bar for preprocessing, if that is even possible
# for evaluating, make a new xMoCoModelPTL with stripped-down functionality and use the test_dataloader to load the corresponding subsets. note that we only have to forward the code, not the docstrings

from appJar import *

app=gui("CodeSearch Tool", "1200x600")
app.setPadding([5, 5])
app.setSticky("ew")
#app.setSticky("news")
#app.setExpand("both")
app.setFont(14)

app.addLabelEntry("Query: ", 0, 0, 1)

def ok(button):
    if button=="OK":
        app.clearTextArea("outputs")
        #app.setTextArea("outputs", "this is \n a text \n\n by me\t and \t him")
        query = app.getEntry("Query: ")
        language = app.getOptionBox("Language: ").lower()
        num_samples = app.getScale("Number of outputs: ")
        subsets = app.getProperties("Subsets to search in: ")
        print(query, language, num_samples, subsets)
        after_str = "#############################################################################\n\n"
        st="def save_act(self, path=None):\n        \"\"\"Save model to a pickle located at `path`\"\"\"\n        if path is None:\n            path = os.path.join(logger.get_dir(), \"model.pkl\")\n\n        with tempfile.TemporaryDirectory() as td:\n            save_variables(os.path.join(td, \"model\"))\n            arc_name = os.path.join(td, \"packed.zip\")\n            with zipfile.ZipFile(arc_name, 'w') as zipf:\n                for root, dirs, files in os.walk(td):\n                    for fname in files:\n                        file_path = os.path.join(root, fname)\n                        if file_path != arc_name:\n                            zipf.write(file_path, os.path.relpath(file_path, td))\n            with open(arc_name, \"rb\") as f:\n                model_data = f.read()\n        with open(path, \"wb\") as f:\n            cloudpickle.dump((model_data, self._act_params), f)"
        app.setTextArea("outputs", "1: (similarity: xxx, match probability: xxx, repo path: xxx)\n\n"+st+"\n\n"+after_str)

app.addButton("OK", ok, 0, 1, 1)
app.addLabelOptionBox("Language: ", ["Ruby", "JavaScript", "Java", "Go", "PHP", "Python"], 0, 2, 1)

subset = {"Train": True, "Validation": True, "Test": True}
app.addProperties("Subsets to search in: ", subset, 1, 2, 1)

app.addLabelScale("Number of outputs: ", 2, 2, 1)
app.setScaleRange("Number of outputs: ", 1, 20)
app.showScaleIntervals("Number of outputs: ", 19)
app.showScaleValue("Number of outputs: ")

app.addScrolledTextArea("outputs", 1, 0, 2, 2)

app.go()

app.go()