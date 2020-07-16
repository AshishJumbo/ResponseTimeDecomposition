<h3 ><em>Response Time Decomposition (RTD)</em></h3>
Exploring the response time decomposition of student action logs in ASSISTments.<br>
Collaborators:<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
AB<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
AG

[link to the data]("https://drive.google.com/drive/folders/1fRhyVEetIsgRdp-B8J5seH64FCHC2HMI?usp=sharing")
[NOTE: there are two files one preprocessed one regular]
<h4>The following is the order of execution of the files in the project:</h4>
<ol>
    <li>libreoffice_prep.py <br/>
        This is the first code base that sorts the data and ensures that everything is in 
        order and all the additional features generation is automated. This takes the 
        <b>...random_sample_20K.csv</b> data and outputs a 
        <b>RTD_data_randomsample_20K.csv</b> data.
        <br>
        Once the RDT_data_randomsample_20K.csv is generated the using libre office to 
        generate the feature values is the quicker option.
        The preprocessing in python is taking forever so had to figure out if it made more sense to have it done in LibreOffice.
        <br>
        <br>
        <hr>
        <ol>
            <li>Make sure to run the libreoffice_prep.py beforehand <br>
                Run the preprocess to clean the PR and PS columns along with the pair features.<br>
            </li>
            <li>Generate action_action_pairs <br> 
                <em>This pairs the relevant actions made by a user per problem to generate the action 
                pairs associated with user made to solve the problem.</em><br>
                Formula: <br>
                =IF(M2 = -1, K2, CONCAT(K2, "_", K3))<br>
                column M : pr<br>
                column K : action_type
            </li>
            <li>Generate action_action_pairs_time_taken <br>
                <em>This calculates the time taken by a user for each action pair while solving the problem.</em><br>
                Formula: <br>
                <!--=IF(M2 = -1, 0, IF(M2 <>M3 , 0,G3 - G2))<br>
                column M : pr<br>-->
                =ROUND(IF(L2 <> L3, 0, G3 - G2), 4) <br>
                column L : ps<br>
                column G : action_unix_time [1 second = 1 unix time]        
            </li>
            <li>Generate pr_answered_correctly_pair <br>
                <em>This checks if the action pair lead to a correct answer to the pr.</em><br>
                Formula: <br>
                <!-- =IF(OR( N5 = "UserSelectedContinueAction", N5 = "ProblemSetMasteredAction" ,  N6 = "UserSelectedContinueAction", N6 = "ProblemSetMasteredAction") , 1, 0) -->
                =IF(N2="StudentResponseAction_ProblemFinishedAction", 1, 0)<br>
                This one is better: <br>
                =IF(OR(N6="StudentResponseAction_ProblemFinishedAction", N7="StudentResponseAction_ProblemFinishedAction"), 1, 0)<br>
                column N: action_action_pairs 
            </li>
            <li>Generate attempts made per Problem: <br>
                <em>This generates all the attemps a student made inorder to answer the pr.</em><br>
                Formula: <br>
                =IF(M2=M3, IF(K2="StudentResponseAction", Q1+1, Q1), 0)<br>
                column M: pr <br>
                column K: action_type<br>
                column Q: number_of_attempts made in the problem
            </li>
            <li>Generate hint requested per Problem:<br>
                <em>This generates all the attemps a student made inorder to answer the pr.</em><br>
                Formula: <br>
                =IF(M2=M3, IF(K2="HintRequestedAction", R1+1, R1), 0)<br>
                column M: pr <br>
                column K: action_type<br>
                column R: number_of_hints accessed in the problem
            </li>
        </ol>
        <hr>
    </li>
    <li>Run the exploratory_data_analysis.py<br/>
        This analysis is still at it's infancy. 
    </li>
</ol>
