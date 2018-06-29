# linking-problem-code
Code related to verb class learning with the Linking Problem

### Readme file for
    (1) predicate_learner python code which is used to
    implement various theories of solving the linking problem as
    measured by the ability to form verb classes ("Predicate learner"
    therefore refers to verb learning) and (2) the Tolerance Principle
    analysis code (derived-linking-tolp/get_pattern_counts.pl) which
    is used to do a link-based Tolerance Principle analysis on linking
    pattern data available in the input-representations subdirectory.

***Updated by Lisa Pearl, 6/29/18

***If using this code, please cite one or more of the following papers:
Pearl, Lisa & Sprouse, Jon. 2018 manuscript. Comparing solutions to
the linking problem using an integrated quantitative framework of
language acquisition. University of California, Irvine and University
of Connecticut, Storrs.

Pearl, Lisa & Sprouse, Jon. 2018 manuscript. The acquisition of linking theories: A Tolerance Principle approach to learning UTAH and rUTAH. University of California, Irvine and University of Connecticut, Storrs.

***Note: The predicate_learner code requires the numpy and scipy python libraries.

****************************
predicate_learner code (learning verb classes)
****************************
(1) Basic usage
Put the predicate-learner.py file in a directory of your choice. Navigate to that directory and run the code. If your input file is the predicates.sampleinput file, you could do it this way from the command line and get the default options.

python predicate_learner.py 'predicates.sampleinput'

You can use the -h option to view a help file that describes all the different options and their default values.

python predicate_learner.py -h

Options:
positional arguments:

input_file            input file containing predicate stats (formatted like predicate.sampleinput)

optional arguments:

-h, --help            show this help message and exit

##### output & debug files #####

--output_file OUTPUT_FILE, -o OUTPUT_FILE

output file name

--debug_file DEBUG_FILE, -df DEBUG_FILE

debug file name

--debug_level DEBUG_LEVEL, -dl DEBUG_LEVEL

debug level 0 to 10000 (the higher the number, the more debug messages are printed out)

##### model variants ######

--with_UTAH, -U       use model that assumes UTAH (+expmapping), so things are interpreted in terms of +/-movement

**note that UTAH vs. rUTAH and +/-surface morphology are implemented during input preprocessing

  ##### hyperparameter: verb classes #####

--gamma_c GAMMA_C, -g_c GAMMA_C

hyperparameter for categories (= verb classes)

##### hyperparameter binary property: animacy of syntactic Subject, Object, and Oblique Object #####

--beta0_anim_gram_subj BETA0_ANIM_GRAM_SUBJ, -b0_ags BETA0_ANIM_GRAM_SUBJ

hyperparameter for binary -anim_gram_subj

--beta1_anim_gram_subj BETA1_ANIM_GRAM_SUBJ, -b1_ags BETA1_ANIM_GRAM_SUBJ

hyperparameter for binary +anim_gram_subj

--beta0_anim_gram_obj BETA0_ANIM_GRAM_OBJ, -b0_ago BETA0_ANIM_GRAM_OBJ

hyperparameter for binary -anim_gram_obj

--beta1_anim_gram_obj BETA1_ANIM_GRAM_OBJ, -b1_ago BETA1_ANIM_GRAM_OBJ

hyperparameter for binary +anim_gram_obj

--beta0_anim_gram_iobj BETA0_ANIM_GRAM_IOBJ, -b0_agi BETA0_ANIM_GRAM_IOBJ

hyperparameter for binary -anim_gram_iobj

--beta1_anim_gram_iobj BETA1_ANIM_GRAM_IOBJ, -b1_agi BETA1_ANIM_GRAM_IOBJ

hyperparameter for binary +anim_gram_iobj

  #### hyperparameter binary property: +/- movement when +expmapping (UTAH) option selected #####

--beta0_mvmt BETA0_MVMT, -b0_m BETA0_MVMT

hyperparameter for binary -mvmt with UTAH (+expmapping) model

--beta1_mvmt BETA1_MVMT, -b1_m BETA1_MVMT

hyperparameter for binary +mvmt with UTAH (+expmapping) model

  #### hyperparameter multinomial property: syntactic frames #####

--alpha_frames ALPHA_FRAMES, -a_f ALPHA_FRAMES

hyperparameter for multinomial syntactic frames distribution

  #### hyperparameter multinomial property: syntactic distribution of "semantic" subject/object/indirect object,

which correspond to proto-AGENT/proto-PATIENT/other for UTAH and HIGHEST/2nd-HIGHEST/3rd-HIGHEST for rUTAH #####

--alpha_sem_subj ALPHA_SEM_SUBJ, -a_ss ALPHA_SEM_SUBJ

hyperparameter for multinomial sem_subj distribution

--alpha_sem_obj ALPHA_SEM_OBJ, -a_so ALPHA_SEM_OBJ

hyperparameter for multinomial sem_obj distribution

--alpha_sem_iobj ALPHA_SEM_IOBJ, -a_si ALPHA_SEM_IOBJ

hyperparameter for multinomial sem_iobj distribution

  ##### simulated annealing options #####

--with_annealing, -A  do annealing

--start_temp START_TEMP, -st_t START_TEMP

starting temperature for annealing

--stop_temp STOP_TEMP, -stp_t STOP_TEMP

stopping temperature for annealing

  ##### sampling options #####

--iterations ITERATIONS, -it ITERATIONS

number of iterations to sample

--sample_hyper, -sam_h

turn on hyperparameter sampling

--eval_iterations EVAL_ITERATIONS, -ev_it EVAL_ITERATIONS

evaluate every how many iterations

****************************
(2) Usage for model options

+/-expmap: 

+expmap (+UTAH) = use UTAH flag,

-expmap (-UTAH) = don't set this flag

***Note: This code was written when we conceptualized UTAH as the mapping rather than the thematic role categorization. That's why "UTAH" in the code corresponds to mapping, while AbsCl-baker corresponds to the thematic role categorization. Sorry about that.
	   
UTAH/rUTAH: 

UTAH = use *.AbsCl-baker.* input-representations files (ex: brown-eve-valian.AbsCl-baker.thresh5.stats)

rUTAH = use *.RelCl.* input-representations files  (ex: brown-eve-valian.RelCl.thresh5.stats)

+/-surfmor: 

+surfmor = use files without *condensedV* (ex: brown-eve-valian.RelCl.thresh5.stats)

-surfmor = use files with *condensedV* (ex: brown-eve-valian.RelCl.condensedV.thresh5.stats)

Example calls:

-expmap, UTAH, +surfmor

python predicate_learner.py 'input-representations/brown-eve-valian/brown-eve-valian.AbsCl-baker.thresh5.stats' -o 'output/brown-eve-valian/-UTAH/AbsCl-baker/brown-eve-valian.AbsCl-baker.thresh5.iter3000.run1.out' -df 'debug/brown-eve-valian/brown-eve-valian.thresh5.debug' -A -sam_h -it 3000

+expmap, rUTAH, -surfmor

python predicate_learner.py 'input-representations/brown-eve-valian/brown-eve-valian.RelCl.condensedV.thresh5.stats' -o 'output/brown-eve-valian/+UTAH/RelCl.condensedV/brown-eve-valian.RelCl.condensedV.thresh5.iter3000.run4.out' -df 'debug/brown-eve-valian/brown-eve-valian.RelCl.condensedV.thresh5.debug' -A -sam_h -U -it 3000

**Note that all the age-appropriate input files derived from the CHILDES Treebank (Pearl & Sprouse 2013) are provided in the input-representations folder.

<3yrs = brown-eve-valian

<4yrs = brown-eve-adam3to4-valian

<5yrs = brown-eve-adam3up-valian


Pearl, L. & Sprouse, J. 2013. Syntactic islands and learning biases: Combining experimental syntax and computational modeling to investigate the language acquisition problem. Language Acquisition, 20, 23-68. DOI 10.1080/10489223.2012.738742.


****************************
(3) Output file information

(a) Initialization values are shown for hyperparameters, whether annealing was used, how many samples are done, how often the state of the model is printed out (Evaluate every XX iterations), and whether hypoerparameters were sampled

(b) At beginning and at each evaluation iteration

After reading in data, total predicates =  XX and total categories = XX

Initial log posterior = -XXXXX.XXXXX

After iteration 1, category stats:

Current number of categories = XX

Categories that changed members = XX

Current log posterior = -XXXXX.XXXXX

(c) After all sampling is complete

After all $num_iterations iterations:

Current number of categories = XXX

Categories that changed members = XX

Current log posterior = -XXXXX.XXXX

****category internal information

Categories look like this:

Category number = X has N predicates:

predicate: PRED1

predicate: PRED2

...

predicate: PREDN

**** category animacy distribution information in Subject, Object, and Obliaue Object positions

anim_gram_subj vs. inanim: XXX.X XXX.X

anim_gram_obj vs. inanim: XXX.X XXX.X

anim_gram_iobj vs. inanim: XXX.X XXX.X

****category thematic role distribution information in Subject, Object and Oblique Object positions

****sem_subj = proto-Agent/Highest

****sem_obj = proto-Patient/2nd-Highest

****sem_iobj = Other/3rd-Highest 

sem_subj as gram_subj vs. gram_obj vs. gram_iobj: XXX.X XXX.X XXX.X

sem_obj as gram_subj vs. gram_obj vs. gram_iobj: XXX.X XXX.X XXX.X

sem_iobj as gram_subj vs. gram_obj vs. gram_iobj: XXX.X XXX.X XXX.X

****category syntactic frame frequencies

****Note: Only the frequencies are listed for each class -- to see which frame corresponds to which line, go to the end of the output file, which will list out the syntactic frames in order.

frame frequencies:

0.000

0.000

0.000

16.000

35.000

214.000

29.000

210.000

0.000

.....(at end of output file)

all frames list:

Item 0: NP Aux V-en

Item 1: NP V

Item 2: NP V CP-null

Item 3: NP V CP-that

Item 4: NP V IP-TO

Item 5: NP V NP

Item 6: NP V NP IP-TO

Item 7: TO V NP

Item 8: TO V PP


****************************
get_pattern_counts.pl (link-based Tolerance Principle analysis for
linking patterns)
****************************
 (1) takes in an input file encoding different properties of verb use within a corpus (formatting described below)
     --- NOTE: Will ignore any verbs whose linking pattern usages are below a certain threshold (default 5 total usages)
 (2) Outputs information related to whether these verbs follow certain linking patterns
     (a) how many total verbs there are
     (b) what the Tolerance Principle (TolP) threshold would be for that many verbs
     (c) how many total verb instances there are 
     (d) what the total counts are for sem-xxbj in gram-xxbj position, across all verbs
     (e) how many verbs of the $total_verb_count total don't follow the primary pattern
     (f) a list of which verbs follow the primary pattern (based on the Tolerance Principle) and which don't
     (g) the analysis for individual links of the primary pattern:
         (i) how many total verbs there are with >=threshold uses for link
         (ii) what the TolP threshold would be for that many verbs
         (iii) how many verbs of the $total_link_verb_count don't follow that link
         (iv) a list of which verbs follow that primary link (based on the Tolerance Principle) and which don't


# input format
# (such as for brown-eve-valian.AbsCl-baker.thresh5.stats or any other file created to be used as input to the linking pattern predicate-learner python code at https://github.com/lisapearl/linking-problem-code)
# verbs are listed out in capitals with accompanying information following
ANSWER
anim-gram-subj vs. inanim: 7 1
anim-gram-obj vs. inanim: 0 8
anim-gram-iobj vs. inanim: 0 0
sem-subj as gram-subj vs. gram-obj vs. gram-iobj: 7 0 0
sem-obj as gram-subj vs. gram-obj vs. gram-iobj: 1 8 0
sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: 0 0 0
frames:
NP VB: 1
...
***
ASK
...


# Output (full file)
 Total verbs: $total_verb_count
 TolP verb threshold: $tolp_threshold
 Verbs not following primary pattern: $not_primary_verbs
 Total verb usage instances: $total_instances
 Counts for total verb usage instances:
 sem-subj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
 sem-obj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
 sem-iobj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+

 verbs not following the primary pattern:
 $not_primary_verb1, $not_primary_verb2, ...

 verbs following the primary pattern:
 $primary_verb1, $primary_verb2,  ...
 ************
 Link: sem-subj as gram-subj

 Total link verbs >= threshold: $total_link_verb_count
 TolP verb threshold: $tolp_link_threshold
 Verb count not following primary link: $not_primary_link_verb_count
 Verbs not following primary link: $not_primary_link_verbs
 Verbs following the primary link: $primary_link_verbs

 ***********
 Link: gram-subj as sem-subj
 ....

 (will also include sem-obj as gram-obj, gram-obj as sem-obj, semi-iobj as gram-iobj, and gram-iobj as semi-iobj)


# Example Usage ##
 get_pattern_counts.pl -input_file "./input-representations/brown-eve-valian/brown-eve-valian.AbsCl-baker.thresh5.stats" -output_file "./output/brown-eve-valian/brown-eve-valian.AbsCl-baker.counts" -threshold 5
