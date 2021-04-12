"""
File that will be responsible of creating the experiment recap: an html page which will contain all the information
 about all the experiments
"""

import os
import json


def create_experiments_recap(path):

    """
    Function that will create an HTML page containing all the information about the different splits
     of our experiments

    :param path: The path to the recordings Folder, the folder that contains all the data about our experiments

    """

    assert os.path.exists(path), "Recordings Folder not found"

    hyperparams_importance_file = "hyperparameters_importance.html"
    parallel_coordinate_file = "parallel_coordinate.html"
    optimization_history_file = "optimization_history.html"

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
    width: fit-content;
    min-width: 100%;
}
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

    body = ""
    board = ""
    evaluation_sections = ""

    for i, evaluation in enumerate(evaluations):
        board += f"""<div class="subtitle center {"subtitle-active" if i==0 else None}">{evaluation}</div>"""

        # We get the folders of all the splits
        splits = os.listdir(os.path.join(path, evaluation))
        splits = [folder for folder in splits if os.path.isdir(os.path.join(path, evaluation, folder))]

        split_board = f"""<div class="split center {"split-active" if i==0 else None}">General</div>"""

        # We open the json file containing the general information of an evaluation
        with open(os.path.join(path, evaluation, "general.json"), "r") as read_file:
            general_data = json.load(read_file)

        main_metrics = ""
        for key in general_data["metrics"].keys():
            main_metrics += f"""
                    <div class="metric-section col center" style="text-align:center">
                        <div class="label">
                            {key}
                        </div>
                        <div class="info">
                            {general_data["metrics"][key]["info"]}
                        </div>
                    </div>
                """

        # We add the general section
        mains = f"""<div class="main {"hidden" if i != 0 else None}" id="{evaluation}General">
            <div class="metrics col center bottom-space">
                {main_metrics}
            </div>
            <img src={os.path.join(path, evaluation,"hyperparameters_importance_recap.png")} >
        </div>"""

        for j, split in enumerate(splits):
            split_board += f"""<div class="split center">{split}</div>"""

            # We open the json file containing information of a split
            with open(os.path.join(path, evaluation, split, "records.json"), "r") as read_file:
                data = json.load(read_file)
            # We add the intro section
            intro = f"""<div class="intro row bottom-space">
                    <div class="intro-section row center">
                        <div class="intro-label">Evaluation name :</div>
                        <div class="intro-info">{evaluation}</div>
                    </div>
                    <div class="intro-section row center">
                        <div class="intro-label">Split index :</div>
                        <div class="intro-info">{split}</div>
                    </div>
                </div>"""

            # We add the hyperparameters section
            hyperparams_section = ""

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

            # We add the metrics section
            metric_section = ""
            for key in data["metrics"].keys():
                metric_section += f"""
                    <div class="metric-section col center">
                        <div class="label">
                            {key}
                        </div>
                        <div class="info">
                            {data["metrics"][key]}
                        </div>
                    </div>
                """
            metric_section = f"""
            <div class="metrics col center bottom-space">
                {metric_section}
            </div>
            """

            # We add the hyperparameters importance secction
            hyperparameters_importance_section = f"""
                        <div class="row center bottom-space">
                            <iframe src="{os.path.join(path, evaluation, split, hyperparams_importance_file)}" 
                            style="width: 90%;height: 100vh;"></iframe>
                        </div>
                    """

            # We add the parallel  coordinate graph
            parallel_coordinate_section = f"""
                        <div class="row center bottom-space">
                            <iframe src="{os.path.join(path, evaluation, split, parallel_coordinate_file)}" 
                            style="width: 90%;height: 100vh;"></iframe>
                        </div>
                    """

            # We add the optimization history graph
            optimization_history_section = f"""
                        <div class="row center bottom-space">
                            <iframe src="{os.path.join(path, evaluation, split, optimization_history_file)}" 
                            style="width: 90%;height: 100vh;"></iframe>
                        </div>
                    """

            # We arrange the different sections
            section = f"""<div class="main hidden" id="{evaluation}{split}">
                {intro}   
                {hyperparams_section}
                {metric_section}
                {hyperparameters_importance_section}
                {parallel_coordinate_section}
                {optimization_history_section}
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
    file = open("Experiments_recap.html", "w")
    file.write(body)
    file.close()