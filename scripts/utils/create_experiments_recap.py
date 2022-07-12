"""
Filename: create_experiments_recap.py

Author: Nicolas Raymond

Description: File responsible of creating the experiment recap: an html page which will contain all the information
             about all the experiments

Date of last modification: 2021/11/23
"""
import argparse
import os
import json

from src.training.tuning import Tuner
from src.utils.visualization import EPOCHS_PROGRESSION_FIG


def argument_parser():
    """
    This function defines a parser that enables user to extract experiment recap from split folders
    """
    # Create a parser
    parser = argparse.ArgumentParser(usage='\n python3 create_experiment_recap.py [experiment_folder_path]',
                                     description="Creates an html file with splits details")

    parser.add_argument('-p', '--path', type=str,
                        help='Path of the experiment folder')

    parser.add_argument('-fn', '--filename', type=str, help='Name of the html file containing the recap')

    arguments = parser.parse_args()

    # Print arguments
    print("\nThe inputs are:")
    for arg in vars(arguments):
        print("{}: {}".format(arg, getattr(arguments, arg)))
    print("\n")

    return arguments


def create_experiments_recap(path: str,
                             filename: str) -> None:
    """
    Creates an HTML page containing all the information about the different splits
    of the experiments

    Args:
        path: path to the folders that contains all the data about the experiments
        filename: name of the html in which we will store the recap

    Returns: None
    """

    if not os.path.exists(path):
        raise ValueError("Recordings Folder not found")

    # We define the style of our webpage with css
    style = """<style>
        body{
    margin: 0px;
    padding: 0px;
    font-family: 'Barlow Condensed', sans-serif;
}

.board{
    z-index: 1;
    position: fixed;
    top: 0px;
    width: 25%;
    height: 100vh;
    background-color: #111;
    color: white;
}
.content{
    position: relative;
    left: 25%;
    height: 100vh;
    width: 75%;
}
.center{
    display: flex;
    justify-content: center;
    align-items: center;
}
.row{
    display: flex;
    flex-direction: row;
}
.col{
    display: flex;
    flex-direction: column;
}

.title-container{
    height: 25vh;
    width: 100%;
    border: 1px white solid;
}
.subtitle-container{
    height: 75vh;
    width: 100%;
    align-items: center;
    overflow: auto ;
}
.title{
    font-size: 2.4em;
}
.subtitle{
    font-size: 1.8em;
    width: 100%;
    height:80px;
    cursor: pointer;
    transition: .3s;
}
.subtitle:hover{
    background-color: grey;
}
.subtitle-active{
    background-color: white;
    color: black;
}
.split-container{
    background-color: #d5dce6;
    height: 8vh;
    font-size: 1.4em;
x   }
.split{
    width: 150px;
    height: 100%;
    cursor: pointer;
}
.split-active{
    background-color: #ebf3ff;
}
.main{
    height: fit-content;
    min-height: 92vh;
}

.bottom-space{
    margin-bottom: 6vh;
}
.intro{
    padding: 20px;
}
.intro-section{
    flex: 1;
}

.intro-label{
    font-size: 1.5em;
    padding-right: 20px;
}
.intro-info{
    font-size: 1.8em;
    font-weight: 600;
}

.hyperparam-container{
    width: 90%;
    flex-wrap: wrap;
    justify-content: center;
}
.hyperparam-section{
    display: flex;
    flex-direction: column;
    width: 30%;
    background-color: #ebf3ff;
    color: black;
    padding:20px 0px;
    margin: 15px 15px

}
.radius{
    border-radius:3px ;
}
.label{
    font-size: 1.6em;
    padding-bottom: 10px;
}
.info{
    font-size: 2.2em;
    font-weight: 600;
}
.metric-section{
    width: 70%;
    background-color: #ebf3ff;
    color: black;
    padding:20px 0px;
    margin: 20px 15px
}
.hidden{
    display: none;
}</style>
    """

    # We define the behaviour of our webpage with javascript
    script = """
    <script>
            const evaluationsBtns = document.querySelectorAll(".subtitle")
            const evaluationsSections = document.querySelectorAll(".content")
            let activeEvaluationBtn = document.querySelector(".subtitle-active")
            let activeSplitBtn = document.querySelector(".split-active")
            for (let i=0;i<evaluationsBtns.length;i++){
                let evaluationsBtn = evaluationsBtns[i]
                evaluationsBtn.addEventListener("click", ()=>{
                    document.getElementById(activeEvaluationBtn.innerText+activeSplitBtn.innerText).classList.toggle("hidden")
                    activeEvaluationBtn.classList.toggle("subtitle-active")
                    document.getElementById(activeEvaluationBtn.innerHTML).classList.toggle("hidden")
                    evaluationsBtn.classList.toggle("subtitle-active")
                    activeEvaluationBtn = evaluationsBtn
                    document.getElementById(activeEvaluationBtn.innerHTML).classList.toggle("hidden")
                    let newSplitBtn = document.getElementById(activeEvaluationBtn.innerHTML).querySelector(".split")
                    activeSplitBtn.classList.toggle("split-active")
                    activeSplitBtn = newSplitBtn
                    activeSplitBtn.classList.toggle("split-active")
                    document.getElementById(activeEvaluationBtn.innerText+activeSplitBtn.innerText).classList.toggle("hidden")
                })
            }

            const splitsBtns = document.querySelectorAll(".split")
            for(let i=0; i<splitsBtns.length;i++){
                let splitBtn = splitsBtns[i]
                splitBtn.addEventListener("click",()=>{
                    activeSplitBtn.classList.toggle("split-active")
                    document.getElementById(activeEvaluationBtn.innerText+activeSplitBtn.innerText).classList.toggle("hidden")
                    activeSplitBtn = splitBtn
                    activeSplitBtn.classList.toggle("split-active")
                    document.getElementById(activeEvaluationBtn.innerText+activeSplitBtn.innerText).classList.toggle("hidden")
                })
            }

        </script>
    """

    # We get the the folders of each evaluation
    evaluations = os.listdir(os.path.join(path))
    evaluations = [folder for folder in evaluations if os.path.isdir(os.path.join(path,folder))]
    evaluations.sort()
    board = ""
    evaluation_sections = ""

    for i, evaluation in enumerate(evaluations):
        board += f"""<div class="subtitle center {"subtitle-active" if i==0 else None}">{evaluation}</div>"""

        # We get the folders of all the splits
        splits = os.listdir(os.path.join(path, evaluation))
        splits = [folder for folder in splits if os.path.isdir(os.path.join(path, evaluation, folder))]
        splits.sort(key=lambda x: int(x.split("_")[1]))

        split_board = f"""<div class="split center {"split-active" if i==0 else None}">General</div>"""

        # We open the json file containing the general information of an evaluation
        with open(os.path.join(path, evaluation, "summary.json"), "r") as read_file:
            general_data = json.load(read_file)

        main_metrics = ""
        for key in general_data["test_metrics"].keys():
            main_metrics += f"""
                    <div class="metric-section col center" style="text-align:center">
                        <div class="label">
                            {key}
                        </div>
                        <div class="info">
                            {general_data["test_metrics"][key]["info"]}
                        </div>
                    x</div>
                """

            main_image = f"""
                    <img src={os.path.join(path, evaluation,"hyperparameters_importance_recap.png")} >
                """ if os.path.exists(os.path.join(path, evaluation, "hyperparameters_importance_recap.png")) else ""

        # We add the general section
        mains = f"""<div class="main {"hidden" if i != 0 else None}" id="{evaluation}General">
            <div class="metrics col center bottom-space">
                {main_metrics}
            </div>
            <div class="metrics col center bottom-space">
            {main_image}
            </div>

        </div>"""

        for j, split in enumerate(splits):
            split_board += f"""<div class="split center">{split}</div>"""

            # We open the json file containing information of a split
            with open(os.path.join(path, evaluation, split, "records.json"), "r") as read_file:
                data = json.load(read_file)

                data_train_info = f"""<div class="intro-section row center">
                            <div class="intro-label">Train set:</div>
                            <div class="intro-info">{data["data_info"]["train_set"]}</div>
                        </div>""" if "train_set" in data["data_info"].keys() else ""

                data_test_info = f"""<div class="intro-section row center">
                            <div class="intro-label">Test set:</div>
                            <div class="intro-info">{data["data_info"]["test_set"]}</div>
                        </div>""" if "test_set" in data["data_info"].keys() else ""

                data_valid_info = f"""<div class="intro-section row center">
                            <div class="intro-label">Valid set:</div>
                            <div class="intro-info">{data["data_info"]["valid_set"]}</div>
                        </div>""" if "valid_set" in data["data_info"].keys() else ""
            # We add the intro section
            intro = f"""
                    <div class="intro row bottom-space">
                        <div class="intro-section row center">
                            <div class="intro-label">Evaluation name :</div>
                            <div class="intro-info">{evaluation}</div>
                        </div>
                        <div class="intro-section row center">
                            <div class="intro-label">Split index :</div>
                            <div class="intro-info">{split}</div>
                        </div>
                    </div>
                    <div class="intro row bottom-space">
                    {data_train_info}
                    {data_valid_info}
                    {data_test_info}
                    </div>
                    """

            # We add the hyperparameters section
            hyperparams_section = ""
            if "hyperparameters" in data.keys():
                for key in data["hyperparameters"].keys():
                    hyperparams_section += f"""
                        <div class="hyperparam-section center">
                            <div class="label">{key}</div>
                            <div class="info">{data["hyperparameters"][key]}</div>
                        </div>
                    """

                hyperparams_section = f"""<div class="hyperparams row center bottom-space">
                        <div class="hyperparam-container row">
                        {hyperparams_section}
                        </div>
                    </div>"""
            else:
                hyperparams_section=""

            # We add the metrics section
            metric_section = ""
            for key in data["test_metrics"].keys():
                metric_section += f"""
                    <div class="metric-section col center">
                        <div class="label">
                            {key}
                        </div>
                        <div class="info">
                            {data["test_metrics"][key]}
                        </div>
                    </div>
                """
            metric_section = f"""
            <div class="metrics col center bottom-space">
                {metric_section}
            </div>
            """

            # We add the hyperparameters importance section
            hyperparameters_importance_section = f"""
                        <div class="row center bottom-space">
                            <img width="1200" src="{os.path.join(path, evaluation, split, Tuner.HPS_IMPORTANCE_FIG)}" 
                            >
                        </div>
                    """ if os.path.exists(os.path.join(path, evaluation, split, Tuner.HPS_IMPORTANCE_FIG)) else ""

            # We add the parallel  coordinate graph
            parallel_coordinate_section = f"""
                        <div class="row center bottom-space">
                            <img src="{os.path.join(path, evaluation, split, Tuner.PARALLEL_COORD_FIG)}" 
                            >
                        </div>
                    """ if os.path.exists(os.path.join(path, evaluation, split, Tuner.PARALLEL_COORD_FIG)) else ""

            # We add the optimization history graph
            optimization_history_section = f"""
                        <div class="row center bottom-space">
                            <img src="{os.path.join(path, evaluation, split, Tuner.OPTIMIZATION_HIST_FIG)}" 
                            ></img>
                        </div>
                """ if os.path.exists(os.path.join(path, evaluation, split, Tuner.OPTIMIZATION_HIST_FIG)) else ""

            # We add the predictions section
            predictions_section = f"""
                        <div class="row center bottom-space">
                            <img width="1200" src="{os.path.join(path, evaluation, split, f"comparison_{evaluation}.png")}">
                        </div>
            """ if os.path.exists(os.path.join(path, evaluation, split, f"comparison_{evaluation}.png")) else ""

            # We add the section visualizing the progression of the loss over the epochs
            loss_over_epochs_section = f"""
                                    <div class="row center bottom-space">
                                        <img width="800" src="{os.path.join(path, evaluation,
                                                                            split, EPOCHS_PROGRESSION_FIG)}">
                                    </div>
                        """ if os.path.exists(
                os.path.join(path, evaluation, split, EPOCHS_PROGRESSION_FIG)) else ""

            # We arrange the different sections
            section = f"""<div class="main hidden" id="{evaluation}{split}">
                {intro}   
                {hyperparams_section}
                {metric_section}
                {hyperparameters_importance_section}
                {parallel_coordinate_section}
                {optimization_history_section}
                {predictions_section}
                {loss_over_epochs_section}
            </div>
            """

            mains += section

        evaluation_sections += f"""
        <div class="content {"hidden" if i >0 else None}" id={evaluation}>
            <div class="split-container row center">
            {split_board}
            </div>
            {mains}
        </div>
        """

    # we prepare the final html content
    body = f"""
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300&display=swap" rel="stylesheet">
        {style}
    </head>
    <body>
    <div class="board center col">
            <div class="title-container center">
                <div class="title">
                    Experiments Recap
                </div>
            </div>
            <div class="subtitle-container">
            {board}
            </div>
    </div>
    {evaluation_sections}
    {script}
    </body>
    
    """
    
    # We save the html file
    file = open(f"{filename}.html", "w")
    file.write(body)
    file.close()


if __name__ == '__main__':

    # Arguments parsing
    args = argument_parser()
    create_experiments_recap(path=args.path, filename=args.filename)