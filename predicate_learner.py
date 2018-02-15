#!/usr/bin/python
"""
Ideal learner model for predicate clustering,
implementing Gibbs sampling.

Created by Lisa Pearl, begun July 9, 2015.
"""

import argparse # for reading in command line arguments
import re # for doing regular expressions
import scipy # for scipy stuff
from scipy import stats # for doing fast random number selection
from scipy.stats import norm # for doing pdf calculation of normal distributions
import numpy as np # for some other speed-up things
from numpy import * # for fast array stuff
from math import log # for log calculations
import sys # for doing prints to STDOUT
import numpy.ma as ma # for masking numpy array values that we want to ignore
import time # for timing program execution

"""
Outline of code

(0) Initialize hyperparameters and annealing schedule parameters.

(1) Read in predicate entries from appropriate input file (predicates.stats)
    These should have a format like the following:

    HEAR
    anim-gram-subj vs. inanim: 101 1
    anim-gram-obj vs. inanim: 50 50
    anim-gram-iobj vs. inanim: 5 2
    sem-subj as gram-subj vs. gram-obj vs. gram-iobj: 100 2 0  
    sem-obj as gram-subj vs. gram-obj vs. gram-iobj: 2 100 0 
    sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: 0 0 7
    frames: 
    TO V PP: 50
    NP V NP: 40
    NP Aux V-en: 2
    NP V CP-that: 8
    NP V CP-null: 2
    ***

    Use these to populate the appropriate data structures (list of Predicates).

(2) Initialize category assignments for each predicate. Use hyperparameter gamma_c,
    and assign using a Chinese Restaurant Process.
    Prob(class c_j) = (# in c_j already + gamma_c)/(# of predicates total + C*gamma_c)
        where c_j is the category assignment and
              C is the number of categories that currently exist.
    First predicate is assigned to category 0 by default.

(3) Initialize binary property counters and multinomial property counters for each category.
    (Probably do this when doing initial category assignments
     and have list of Category types since need to have running totals of things for categories.)
    Binary properties:
       anim_gram_subj, inanim_gram_subj
       anim_gram_obj, inanim_gram_obj
       anim_gram_iobj, inanim_gram_iobj
       mvmt (if UTAH) 
    Multinomial properties:
       if no UTAH:
         sem_subj (as gram_subj, gram_obj, or gram_ind_obj)
         sem_obj (as gram_subj, gram_obj, gram_ind_obj)
         sem_iobj (as gram_subj, gram_obj, gram_ind_obj)
       frames (as TO V PP, NP V NP, NP Aux V-en, ...)

(4) Set annealing schedule.

(5) For 1 to num_iterations
       For all predicates:
          Sample predicate category, following annealing.
       Sample hyperparameters (following annealing) for
       category (gamma_c),
       binary properties (betas), and
       multinomial properties (alphas).

       Evaluate every XX iterations
         (1) how many category assignments have changed
         (2) log posterior, given current category assignments
            
"""
# debug_val options:
#    10000 or above = turn on every last debug statement

def main():
    # start timing
    start_time = time.time()
    
    # read in relevant argument values from command line
    # required: input file name with predicate statistics (ex: predicates.stats)
    # optional: output file, debug settings, UTAH or no, hyperparameters, annealing schedule parameters
    parser = argparse.ArgumentParser(description="Predicate Learner Gibbs sampler")
    parser.add_argument("--output_file", "-o", type=str, default="predlearner.out", \
                        help="output file name")
    parser.add_argument("--debug_file", "-df", type=str, default="predlearner.debug", \
                        help="debug file name")
    parser.add_argument("--debug_level", "-dl", type=int, default=100, \
                        help="debug level 0 to 10000")
           
    parser.add_argument("--with_UTAH", "-U", action="store_true", \
                        help="use model that assumes UTAH (+expmapping)") 
    # hyperparameter args
    parser.add_argument("--gamma_c", "-g_c", type=float, default=1.0, \
                        help="hyperparameter for categories")
    # binary 
    parser.add_argument("--beta0_anim_gram_subj", "-b0_ags", type=float, default=0.001, \
                        help="hyperparameter for binary -anim_gram_subj")       
    parser.add_argument("--beta1_anim_gram_subj", "-b1_ags", type=float, default=0.001, \
                        help="hyperparameter for binary +anim_gram_subj")  
    parser.add_argument("--beta0_anim_gram_obj", "-b0_ago", type=float, default=0.001, \
                        help="hyperparameter for binary -anim_gram_obj") 
    parser.add_argument("--beta1_anim_gram_obj", "-b1_ago", type=float, default=0.001, \
                        help="hyperparameter for binary +anim_gram_obj") 
    parser.add_argument("--beta0_anim_gram_iobj", "-b0_agi", type=float, default=0.001, \
                        help="hyperparameter for binary -anim_gram_iobj") 
    parser.add_argument("--beta1_anim_gram_iobj", "-b1_agi", type=float, default=0.001, \
                        help="hyperparameter for binary +anim_gram_iobj") 
    parser.add_argument("--beta0_mvmt", "-b0_m", type=float, default=0.001, \
                        help="hyperparameter for binary -mvmt with UTAH model") 
    parser.add_argument("--beta1_mvmt", "-b1_m", type=float, default=0.001, \
                        help="hyperparameter for binary +mvmt with UTAH model")
    # multinomial 
    parser.add_argument("--alpha_frames", "-a_f", type=float, default=0.001, \
                        help="hyperparameter for multinomial frames distribution") 
    parser.add_argument("--alpha_sem_subj", "-a_ss", type=float, default=0.001, \
                        help="hyperparameter for multinomial sem_subj distribution")  
    parser.add_argument("--alpha_sem_obj", "-a_so", type=float, default=0.001, \
                        help="hyperparameter for multinomial sem_obj distribution") 
    parser.add_argument("--alpha_sem_iobj", "-a_si", type=float, default=0.001, \
                        help="hyperparameter for multinomial sem_iobj distribution") 

                                    
    # annealing args
    parser.add_argument("--with_annealing", "-A", action="store_true", \
                        help="do annealing") 
    parser.add_argument("--start_temp", "-st_t", type=float, default=2.0, \
                        help="starting temperature for annealing")
    parser.add_argument("--stop_temp", "-stp_t", type=float, default=0.8, \
                        help="stopping temperature for annealing")

    # iterations, sample hyperparameters or not, eval every XX iterations
    parser.add_argument("--iterations", "-it", type=int, default=100, \
                        help="number of iterations to sample")                    
    parser.add_argument("--sample_hyper", "-sam_h", action="store_true", \
                        help="turn on hyperparameter sampling")
    parser.add_argument("--eval_iterations", "-ev_it", type=int, default=10, \
                        help="evaluate every how many iterations")  
                                                
    parser.add_argument("input_file", type=str, \
                        help="input file containing predicate stats") # required argument
    args = parser.parse_args()
   

    # open debug file for printing out debug stuff
    debug_file = open(args.debug_file, 'w')
    # open output file for printint out output stuff
    output_file = open(args.output_file, 'w')

    if(args.debug_level >= 10000):
        debug_file.write("DEBUG: Starting main code block\n")
            
    # (0) Initialize hyperparameters and annealing schedule parameters.
    this_pred_learner = \
        PredLearner(debug_file, output_file,\
                args.debug_level, args.with_UTAH, args.gamma_c, \
                args.beta0_anim_gram_subj, args.beta1_anim_gram_subj, \
                args.beta0_anim_gram_obj, args.beta1_anim_gram_obj, \
                args.beta0_anim_gram_iobj, args.beta1_anim_gram_iobj, \
                args.beta0_mvmt, args.beta1_mvmt, \
                args.alpha_frames, \
                args.alpha_sem_subj, args.alpha_sem_obj, args.alpha_sem_iobj, \
                args.with_annealing, args.start_temp, args.stop_temp,\
                args.iterations, args.sample_hyper, args.eval_iterations)

    # (1) Read in predicate entries from appropriate input file (predicates.stats)
    this_pred_learner.read_pred_entries(args.input_file, args.debug_level, debug_file)

    # (2) Initialize category assignments for each predicate. Use hyperparameter gamma_c,
    #     and assign using a Chinese Restaurant Process.
    # (3) Initialize binary property counters and multinomial property counters for each category.
    this_pred_learner.initialize_categories(args.debug_level, debug_file)

    # (4) Set annealing schedule if doing annealing
    if args.with_annealing:
        this_pred_learner.set_annealing_schedule(args.debug_level, debug_file, output_file)

    # (5) Sample and evaluate
    this_pred_learner.sample_and_eval(args.debug_level, debug_file, output_file)
    
    # close output file
    output_file.close()
    # close debug file
    debug_file.close()

    if args.debug_level >= 100:
        print("--- {} seconds ---\n".format(time.time() - start_time))

####################################                     
# main class that holds the sampler
class PredLearner:
    def sample_and_eval(self, debug_level, debug_file, output_file):
        self.sample_init(debug_level, debug_file, output_file)         # initialize various things
        # for each iterations in self.iterations
        sys.stdout.write("Sampling predicates.")
        for iter in range(self.iterations):
            if (iter % self.eval_iterations) == 0: # print out a . for every eval_iterations progress
                sys.stdout.write(".")
                sys.stdout.flush() # force it to print this out right away
            if debug_level >= 7000:
                debug_file.write("Iteration {}\n".format(iter))
            # if annealing, check to see if we need to update temperature
            if self.with_annealing:
                self.get_curr_temp(debug_level, debug_file, iter)
                   
            self.sample_predicates(debug_level, debug_file)         # sample each predicate in self.predicates

            # --> recompute self.cat_prob_logs for existing hyperparameters & then compute for new hyperparameters
            self.calc_all_cat_prob_logs(debug_level, debug_file)
            # sample hyperparameters & update if necessary
            if self.sample_hyper:
                if debug_level >= 3000:
                    debug_file.write("Sampling hyperparameters for iteration {}\n".format(iter))
                self.sample_hyperparameters(debug_level, debug_file)    
        
            # see if time to print out eval stats (log prob, how many categories changed assignments, etc.)
            #    based on self.eval_iterations
            if iter % self.eval_iterations == 0:
                output_file.write("After iteration {}, category stats:\n".format(iter+1))
                self.print_learner_stats(debug_level, debug_file, output_file)
            # --> then reset self.category_changed
            self.category_changed = np.zeros((len(self.cat_freq)))
            
        print("") # close off sampling iter line of ....
        output_file.write("After all {} iterations:\n".format(self.iterations))
        self.print_learner_stats(debug_level, debug_file, output_file)
        output_file.write("Categories look like this:\n")
        for cat_index in range(len(self.categories)):
            if self.categories[cat_index].num_preds: # make sure not deleted category
                self.categories[cat_index].print_cat_info(output_file, self.with_UTAH)
        # print out all_frames so mapping to category frames is clear
        output_file.write("all frames list:\n") #{}".format(self.all_frames))
        print_array(self.all_frames, output_file)

        if debug_level >= 100:
            debug_file.write("After all {} iterations, categories look like this:\n".format(self.iterations))
            for cat_index in range(len(self.categories)):
                if self.categories[cat_index].num_preds: # make sure not deleted category
                    self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)


    def sample_hyperparameters(self, debug_level, debug_file):
        # sample category gamma_c
        self.sample_hyper_gamma(debug_level, debug_file)
        # sample binary property beta parameters
        self.sample_hyper_beta(debug_level, debug_file)
        # sample multinomial property alpha parameters
        self.sample_hyper_alpha(debug_level, debug_file)

        # if self.updated_gamma_hyper or self.updated_beta_hyper or self.updated_alpha_hyper,
        # recalculate self.cat_prob_logs
        if self.updated_gamma_hyper or self.updated_beta_hyper or self.updated_alpha_hyper:
            if debug_level >= 9000:
                debug_file.write("self.updated_alpha_hyper = {}, ".format(self.updated_alpha_hyper))
                debug_file.write("self.updated_beta_hyper = {}, ".format(self.updated_beta_hyper))
                debug_file.write("and self.updated_gamma_hyper = {}.\n".format(self.updated_gamma_hyper))
                debug_file.write("Before updating cat_prob_logs, it is\n")
                np.savetxt(debug_file, self.cat_prob_logs, fmt='%1.9f')
            for cat_index in np.nonzero(self.cat_freq)[0]:
                if cat_index < (len(self.cat_freq)-1): # not empty category
                    self.cat_prob_logs[cat_index] = log(self.cat_freq[cat_index]/self.cat_freq.sum(axis=0)) +\
                                                    self.log_p_binary[cat_index] + self.log_p_multinomial[cat_index]
            if debug_level >= 9000:
                debug_file.write("After updating cat_prob_logs, it is\n")
                np.savetxt(debug_file, self.cat_prob_logs, fmt='%1.9f')

        
    def sample_hyper_alpha(self, debug_level, debug_file):
        # Sample all multinomial hyperparameters one at a time
        #  self.fast_frames_raw_probs, self.fast_frames_probs and
        #  if not self.with_UTAH, sem_XXX_raw_probs, self.sem_XXX_probs

        self.updated_alpha_hyper = False # set to True if any updated, because then need to recalc log_p_multinomial
        # fast_frames
        # create holder for hyperparameter versions of category calculations in case need to replace existing with them
        self.hyper_cat_fast_frames_raw_probs = np.zeros((len(self.cat_freq), len(self.all_frames)))
        self.hyper_cat_fast_frames_probs = np.zeros((len(self.cat_freq), len(self.all_frames)))
        self.sample_hyper_alpha_frames(debug_level, debug_file)

        if not self.with_UTAH:
            # sem_subj, sem_obj, sem_iobj
            self.hyper_cat_sem_subj_raw_probs = np.zeros((len(self.cat_freq), 3))
            self.hyper_cat_sem_subj_probs = np.zeros((len(self.cat_freq), 3))
            self.hyper_cat_sem_obj_raw_probs = np.zeros((len(self.cat_freq), 3))
            self.hyper_cat_sem_obj_probs = np.zeros((len(self.cat_freq), 3))
            self.hyper_cat_sem_iobj_raw_probs = np.zeros((len(self.cat_freq), 3))
            self.hyper_cat_sem_iobj_probs = np.zeros((len(self.cat_freq), 3))
            self.sample_hyper_alpha_sem(debug_level, debug_file, \
                                        self.alphas['alpha_sem_subj'], self.alphas['alpha_sem_obj'], \
                                        self.alphas['alpha_sem_iobj'])

        #if self.updated_alpha_hyper, update self.log_p_multinomial
        if self.updated_alpha_hyper:
            if debug_level >= 9000:
                debug_file.write("self.updated_alpha_hyper = True. Before updating log_p_multinomial, it is\n")
                np.savetxt(debug_file, self.log_p_multinomial, fmt='%1.9f')
            for cat_index in np.nonzero(self.cat_freq)[0]:
                if cat_index < (len(self.cat_freq)-1): # not empty category
                    self.log_p_multinomial[cat_index] = self.calc_log_p_multinomial(debug_level, debug_file, cat_index)
            if debug_level >= 9000:
                debug_file.write("After updating log_p_multinomial, it is\n")
                np.savetxt(debug_file, self.log_p_multinomial, fmt='%1.9f')

    def sample_hyper_alpha_sem(self, debug_level, debug_file, \
                               old_alpha_subj, old_alpha_obj, old_alpha_iobj):
        # get_proposal_values
        proposed_alpha_subj = self.get_proposal_value(debug_level, debug_file, old_alpha_subj)
        proposed_alpha_obj = self.get_proposal_value(debug_level, debug_file, old_alpha_obj)
        proposed_alpha_iobj = self.get_proposal_value(debug_level, debug_file, old_alpha_iobj)
        
        if debug_level >= 8000:
            debug_file.write("sample_hyper_alpha_sem, ")
            debug_file.write("subj: old = {}, proposed = {}\n".\
                             format(old_alpha_subj, proposed_alpha_subj))
            debug_file.write("obj: old = {}, proposed = {}\n".\
                             format(old_alpha_obj, proposed_alpha_obj))
            debug_file.write("obj: old = {}, proposed = {}\n".\
                             format(old_alpha_iobj, proposed_alpha_iobj))
                             
        # set hypers appropriately
        self.alphas['alpha_sem_subj_hyper'] = proposed_alpha_subj
        self.alphas['alpha_sem_obj_hyper'] = proposed_alpha_obj
        self.alphas['alpha_sem_iobj_hyper'] = proposed_alpha_iobj

        # now calculate p_old and p_new appropriately by populating appropriate self.hyper_cat_sem_xxx(_raw)_probs
        # p(pred | old/new) = prod (# instances s_j) ^ (p_s_j) => sum(p_s_j * log(# instances s_j))
        for cat_index in np.nonzero(self.cat_freq)[0]: # non-deleted categories & empty category
            self.calc_sem_probs_cat(debug_level, debug_file, cat_index, hyper=True)
        
        # then calculate sum for each category and sum of all categories to get p_old and p_new
        # for both, make sure to subtract off contribution from empty category
        p_old_hyper_subj = self.sem_subj_probs.sum() - self.sem_subj_probs[-1].sum(axis=0)
        p_new_hyper_subj = self.hyper_cat_sem_subj_probs.sum() - self.hyper_cat_sem_subj_probs[-1].sum(axis=0)  
        p_old_hyper_obj = self.sem_obj_probs.sum() - self.sem_obj_probs[-1].sum(axis=0)
        p_new_hyper_obj = self.hyper_cat_sem_obj_probs.sum() - self.hyper_cat_sem_obj_probs[-1].sum(axis=0)
        p_old_hyper_iobj = self.sem_iobj_probs.sum() - self.sem_iobj_probs[-1].sum(axis=0)
        p_new_hyper_iobj = self.hyper_cat_sem_iobj_probs.sum() - self.hyper_cat_sem_iobj_probs[-1].sum(axis=0)  

        if debug_level >= 9000:
            debug_file.write("p_old_hyper_subj = {}\n".format(p_old_hyper_subj))
            debug_file.write("p_new_hyper_subj = {}\n".format(p_new_hyper_subj))
            debug_file.write("p_old_hyper_obj = {}\n".format(p_old_hyper_obj))
            debug_file.write("p_new_hyper_obj = {}\n".format(p_new_hyper_obj))
            debug_file.write("p_old_hyper_iobj = {}\n".format(p_old_hyper_iobj))
            debug_file.write("p_new_hyper_iobj = {}\n".format(p_new_hyper_iobj))
                            
        # note: need to add something to each one to get rid of underflow issues (pick smaller one)
        p_old_hyper_subj, p_new_hyper_subj = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper_subj, p_new_hyper_subj)
        p_old_hyper_obj, p_new_hyper_obj = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper_obj, p_new_hyper_obj)
        p_old_hyper_iobj, p_new_hyper_iobj = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper_iobj, p_new_hyper_iobj)
                
        # then metroplis-hastings sample for each one
        selected_hyper_subj = self.mh_sampling(debug_level, debug_file, p_old_hyper_subj, p_new_hyper_subj, \
                                     self.alphas['alpha_sem_subj'], self.alphas['alpha_sem_subj_hyper'])
        selected_hyper_obj = self.mh_sampling(debug_level, debug_file, p_old_hyper_obj, p_new_hyper_obj, \
                                     self.alphas['alpha_sem_obj'], self.alphas['alpha_sem_obj_hyper'])
        selected_hyper_iobj = self.mh_sampling(debug_level, debug_file, p_old_hyper_iobj, p_new_hyper_iobj, \
                                     self.alphas['alpha_sem_iobj'], self.alphas['alpha_sem_iobj_hyper'])
                                                                                  
        # if selecting new hyper, update self.sem_xxx(_raw)_probs and self.['alpha_sem_XXX'], set self.updated_alpha_hyper = True
        if selected_hyper_subj:
            np.copyto(self.sem_subj_probs, self.hyper_cat_sem_subj_probs)
            np.copyto(self.sem_subj_raw_probs, self.hyper_cat_sem_subj_raw_probs)
            self.alphas['alpha_sem_subj'] = 0 + self.alphas['alpha_sem_subj_hyper']
            if debug_level >= 8000:
                debug_file.write("sample_hyper_alpha_sem_subj, selected_hyper, so sem_subj_probs now\n")
                np.savetxt(debug_file, self.sem_subj_probs, fmt='%1.9f')
                debug_file.write("and sem_subj_raw_probs now\n")
                np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.9f')
                debug_file.write("and alpha_sem_subj now = {}\n".format(self.alphas['alpha_sem_subj']))
            self.updated_alpha_hyper = True
            
        if selected_hyper_obj:
            np.copyto(self.sem_obj_probs, self.hyper_cat_sem_obj_probs)
            np.copyto(self.sem_obj_raw_probs, self.hyper_cat_sem_obj_raw_probs)
            self.alphas['alpha_sem_obj'] = 0 + self.alphas['alpha_sem_obj_hyper']
            if debug_level >= 8000:
                debug_file.write("sample_hyper_alpha_sem_obj, selected_hyper, so sem_obj_probs now\n")
                np.savetxt(debug_file, self.sem_obj_probs, fmt='%1.9f')
                debug_file.write("and sem_obj_raw_probs now\n")
                np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.9f')
                debug_file.write("and alpha_sem_obj now = {}\n".format(self.alphas['alpha_sem_obj']))
            self.updated_alpha_hyper = True
            
        if selected_hyper_iobj:
            np.copyto(self.sem_iobj_probs, self.hyper_cat_sem_iobj_probs)
            np.copyto(self.sem_iobj_raw_probs, self.hyper_cat_sem_iobj_raw_probs)
            self.alphas['alpha_sem_iobj'] = 0 + self.alphas['alpha_sem_iobj_hyper']
            if debug_level >= 8000:
                debug_file.write("sample_hyper_alpha_sem_iobj, selected_hyper, so sem_iobj_probs now\n")
                np.savetxt(debug_file, self.sem_iobj_probs, fmt='%1.9f')
                debug_file.write("and sem_iobj_raw_probs now\n")
                np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.9f')
                debug_file.write("and alpha_sem_iobj now = {}\n".format(self.alphas['alpha_sem_iobj']))
            self.updated_alpha_hyper = True
        
        
    def sample_hyper_alpha_frames(self, debug_level, debug_file):
        # get proposal value
        self.alphas['alpha_frames_hyper'] = self.get_proposal_value(debug_level, debug_file, self.alphas['alpha_frames'])
        if debug_level >= 8000:
            debug_file.write("\nsample_hyper_alpha_frames: proposed value is {}\n".format(self.alphas['alpha_frames_hyper']))
        
        # calculate p(all pred | old) and p(all pred | new) for each category, then sum over all categories
        # --> prod[(# frames of f_j) ^ (p_f_j)] => sum(p_f_j * log(# frames of f_j)
        for cat_index in np.nonzero(self.cat_freq)[0]: # non-deleted categories & empty category
            self.calc_frames_probs_cat(debug_level, debug_file, cat_index, hyper=True)
        if debug_level >= 8000:
            debug_file.write("sample_hyper_alpha_frames: after updating hyper_cat_fast_frames, \n")
            debug_file.write("self.hyper_cat_fast_frames_raw_probs = \n")
            np.savetxt(debug_file, self.hyper_cat_fast_frames_raw_probs, fmt='%1.9f')
            debug_file.write("while self.fast_frames_raw_probs is \n")
            np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.9f')            
            debug_file.write("self.hyper_cat_fast_frames_probs = \n")
            np.savetxt(debug_file, self.hyper_cat_fast_frames_probs, fmt='%1.9f')
            debug_file.write("while self.fast_frames_probs is \n")
            np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.9f')
            
        # p_old = sum over all frames over non-zero and non-empty categories from old frame log probs
        # p_new = sum over all frames over non-zero and non-empty categories from new hyper frame log probs
        # for both, make sure to subtract off contribution from empty category
        p_old_hyper = self.fast_frames_probs.sum() - self.fast_frames_probs[-1].sum(axis=0)  
        p_new_hyper = self.hyper_cat_fast_frames_probs.sum() - self.hyper_cat_fast_frames_probs[-1].sum(axis=0) 
        if debug_level >= 9000:
            debug_file.write("p_old_hyper for fast_frames = {}\n".format(p_old_hyper))
            debug_file.write("p_new_hyper for fast_frames = {}\n".format(p_new_hyper))
        # note: need to add something to each one to get rid of underflow issues (pick smaller one)
        p_old_hyper, p_new_hyper = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper, p_new_hyper)
            
        # do metropolis-hastings sampling to see if select new value
        selected_hyper = self.mh_sampling(debug_level, debug_file, p_old_hyper, p_new_hyper, \
                                     self.alphas['alpha_frames'], self.alphas['alpha_frames_hyper'])
                                     
        if selected_hyper:
            # if so, update self.fast_frames_probs and self.fast_frames_raw_probs
            np.copyto(self.fast_frames_probs, self.hyper_cat_fast_frames_probs)
            np.copyto(self.fast_frames_raw_probs, self.hyper_cat_fast_frames_raw_probs)
            # and then self.alphas['alpha_frames']
            self.alphas['alpha_frames'] = 0 + self.alphas['alpha_frames_hyper']
            if debug_level >= 8000:
                debug_file.write("sample_hyper_alpha_frames, selected_hyper, so fast_frames_probs now\n")
                np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.9f')
                debug_file.write("and fast_frames_raw_probs now\n")
                np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.9f')
                debug_file.write("and alpha_frames now = {}\n".format(self.alphas['alpha_frames']))
                
            # and set self.updated_alpha_hyper = True
            self.updated_alpha_hyper = True
        
        
    def sample_hyper_beta(self, debug_level, debug_file):
        # Sample all binary hyperparameters one set at a time
        # for each binary property

        # create holder for hyperparameter versions of category calculations in case need to replace existing with them
        self.hyper_cat_bin_raw_probs = np.zeros((len(self.cat_freq), 3+self.with_UTAH))
        self.hyper_cat_bin_log_probs = np.zeros((len(self.cat_freq), 3+self.with_UTAH))
        self.updated_beta_hyper = False # set to True if any updated, because then need to recalculate log_p_binary
        # for all: anim_gram_subj, anim_gram_obj, anim_gram_iobj
        # start with anim_gram_subj
        if debug_level >= 9000:
            debug_file.write("sample_hyper_beta: doing anim_gram_subj betas\n")
        self.sample_hyper_beta_indiv(debug_level, debug_file, 0, \
                                     self.betas['beta0_anim_gram_subj'], self.betas['beta1_anim_gram_subj'])
        # then anim_gram_obj
        if debug_level >= 9000:
            debug_file.write("sample_hyper_beta: doing anim_gram_obj betas\n")
        self.sample_hyper_beta_indiv(debug_level, debug_file, 1, \
                                     self.betas['beta0_anim_gram_obj'], self.betas['beta1_anim_gram_obj'])
        # then anim_gram_iobj
        if debug_level >= 9000:
            debug_file.write("sample_hyper_beta: doing anim_gram_iobj betas\n")
        self.sample_hyper_beta_indiv(debug_level, debug_file, 2, \
                                     self.betas['beta0_anim_gram_iobj'], self.betas['beta1_anim_gram_iobj']) 
        # if self.with_UTAH: do mvmt (index 3)
        if self.with_UTAH:
            if debug_level >= 9000:
                debug_file.write("sample_hyper_beta: doing mvmt betas\n")
            self.sample_hyper_beta_indiv(debug_level, debug_file, 3, \
                                        self.betas['beta0_mvmt'], self.betas['beta1_mvmt'])            

        # if self.updated_bin_hyper, recalculate log_p_binary
        # then recalculate self.log_p_binary for non-deleted and non-empty categories
        if self.updated_beta_hyper:
            if debug_level >= 9000:
                debug_file.write("Before updating log_p_binary, it looks like this\n")
                np.savetxt(debug_file, self.log_p_binary, fmt='%1.9f')
            ## Make sure to do only for non-zero entries 
            for cat_index in np.nonzero(self.cat_freq)[0]:
                self.log_p_binary[cat_index] = self.sampling_bin_prop[cat_index].sum()
            if debug_level >= 9000:
                debug_file.write("After updating log_p_binary, it looks like this\n")
                np.savetxt(debug_file, self.log_p_binary, fmt='%1.9f')
        
    def sample_hyper_beta_indiv(self, debug_level, debug_file, bin_prop, old_beta0, old_beta1):
        # get proposal values
        proposed_beta0 = self.get_proposal_value(debug_level, debug_file, old_beta0)
        proposed_beta1 = self.get_proposal_value(debug_level, debug_file, old_beta1)
                
        if bin_prop == 0: # anim_gram_subj
            self.betas['beta0_anim_gram_subj_hyper'] = proposed_beta0
            self.betas['beta1_anim_gram_subj_hyper'] = proposed_beta1
        elif bin_prop == 1: # anim_gram_obj
            self.betas['beta0_anim_gram_obj_hyper'] = proposed_beta0
            self.betas['beta1_anim_gram_obj_hyper'] = proposed_beta1
        elif bin_prop == 2: # anim_gram_iobj
            self.betas['beta0_anim_gram_iobj_hyper'] = proposed_beta0
            self.betas['beta1_anim_gram_iobj_hyper'] = proposed_beta1
        elif bin_prop == 3: # mvmt
            self.betas['beta0_mvmt_hyper'] = proposed_beta0
            self.betas['beta1_mvmt_hyper'] = proposed_beta1         
                        
        if debug_level >= 7000:
            debug_file.write("sample_hyper_beta: current betas for bin_prop index {}:\n".format(bin_prop))
            debug_file.write("beta0: {}, beta1: {}\n".format(old_beta0, old_beta1))
            debug_file.write("sample_hyper_beta: proposed betas for anim_gram_subj:\n")
            debug_file.write("beta0: {}, beta1: {}\n".format(proposed_beta0, proposed_beta1))

        # now calculate p(predicates | old values) vs. p(predicates | proposed values)
        # --> (p_bin)^yes_bin * (1-p_bin)^no_bin = yes_bin*log(p_bin) + no_bin*log(1-p_bin)
        #     for each non-deleted and non-empty category
        if debug_level >= 9000:
            debug_file.write("sample_hyper_beta_indiv: self.cat_freq looks like this:\n")
            np.savetxt(debug_file, self.cat_freq, fmt='%1.4f')
            debug_file.write("Before calculating bin_prop {}, ".format(bin_prop))
            debug_file.write("self.hyper_cat_bin_raw_probs looks like this\n")
            np.savetxt(debug_file, self.hyper_cat_bin_raw_probs, fmt='%1.4f')
            debug_file.write("and self.hyper_cat_bin_log_probs looks like this\n")
            np.savetxt(debug_file, self.hyper_cat_bin_log_probs, fmt='%1.4f')
        for cat_index in np.nonzero(self.cat_freq)[0]:
            if debug_level >= 9000:
                debug_file.write("sample_hyper_beta_indiv: recalculating bin_prop {} stuff for cat_index {}\n".\
                                 format(bin_prop, cat_index))
            if cat_index < len(self.cat_freq)-1:
                self.hyper_cat_bin_raw_probs[cat_index, bin_prop], \
                self.hyper_cat_bin_log_probs[cat_index, bin_prop] = \
                self.calc_sample_bin_prop(debug_level, debug_file, bin_prop, self.categories[cat_index],  \
                                 hyper=True, hyper_beta0=proposed_beta0, hyper_beta1=proposed_beta1)
            else: # last (empty) category stats
                self.hyper_cat_bin_raw_probs[cat_index, bin_prop], \
                self.hyper_cat_bin_log_probs[cat_index, bin_prop] = \
                self.calc_sample_bin_prop(debug_level, debug_file, bin_prop,  \
                                 hyper=True, hyper_beta0=proposed_beta0, hyper_beta1=proposed_beta1)
        if debug_level >= 9000:
            debug_file.write("sample_hyper_beta_indiv: self.cat_freq looks like this:\n")
            np.savetxt(debug_file, self.cat_freq, fmt='%1.4f')
            debug_file.write("After calculating bin_prop {}, ".format(bin_prop))
            debug_file.write("self.hyper_cat_bin_raw_probs looks like this\n")
            np.savetxt(debug_file, self.hyper_cat_bin_raw_probs, fmt='%1.4f')
            debug_file.write("and self.hyper_cat_bin_log_probs looks like this\n")
            np.savetxt(debug_file, self.hyper_cat_bin_log_probs, fmt='%1.4f')
            debug_file.write("and self.sampling_bin_prop looks like this\n")
            np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.4f')            
                                                          
        # and then sum of logs over all non-deleted & non-empty categories
        p_old_hyper = self.sampling_bin_prop[np.nonzero(self.cat_freq)[0], bin_prop].sum(axis=0) \
                      - self.sampling_bin_prop[-1, bin_prop] # ignore contribution from empty category
        p_new_hyper = self.hyper_cat_bin_log_probs[:,bin_prop].sum(axis=0) \
                      - self.hyper_cat_bin_log_probs[-1, bin_prop] # ignore contribution from empty category
        if debug_level >= 9000:
            debug_file.write("p_old_hyper for bin_prop {} = {}\n".format(bin_prop, p_old_hyper))
            debug_file.write("p_new_hyper for bin_prop {} = {}\n".format(bin_prop, p_new_hyper))
        # note: need to add something to each one to get rid of underflow issues (pick smaller one)
        p_old_hyper, p_new_hyper = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper, p_new_hyper)
        
        # do metropolis-hastings sampling to determine if select new values
        if bin_prop == 0: #anim_gram_subj
            kappa = self.betas['beta0_anim_gram_subj']
            kappa_prime = self.betas['beta0_anim_gram_subj_hyper']
            kappa2 = self.betas['beta1_anim_gram_subj']
            kappa2_prime = self.betas['beta1_anim_gram_subj_hyper']
        elif bin_prop == 1: # anim_gram_obj
            kappa = self.betas['beta0_anim_gram_obj']
            kappa_prime = self.betas['beta0_anim_gram_obj_hyper']
            kappa2 = self.betas['beta1_anim_gram_obj']
            kappa2_prime = self.betas['beta1_anim_gram_obj_hyper']
        elif bin_prop == 2: # anim_gram_iobj
            kappa = self.betas['beta0_anim_gram_iobj']
            kappa_prime = self.betas['beta0_anim_gram_iobj_hyper']
            kappa2 = self.betas['beta1_anim_gram_iobj']
            kappa2_prime = self.betas['beta1_anim_gram_iobj_hyper']
        elif bin_prop == 3: # mvmt
            kappa = self.betas['beta0_mvmt']
            kappa_prime = self.betas['beta0_mvmt_hyper']
            kappa2 = self.betas['beta1_mvmt']
            kappa2_prime = self.betas['beta1_mvmt_hyper']
        
        selected_hyper = self.mh_sampling(debug_level, debug_file, p_old_hyper, p_new_hyper, \
                                     kappa, kappa_prime, kappa2, kappa2_prime)
        # if so, update dependent binary properties (sampling_bin_raw_prop, sampling_bin_prop)
        if selected_hyper: # selected index 1 instead of 0
            if debug_level >= 9000:
                debug_file.write("selected new hyper for bin_prop {}: Need to update with new betas beta0 = {} and beta1 = {}\n".\
                                 format(bin_prop, proposed_beta0, proposed_beta1))
                debug_file.write("Before updating self.sampling_bin_prop, it looks like this\n")
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.9f')
                debug_file.write(" and self.sampling_bin_raw_prop looks like this\n")
                np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.9f')
                                 
            # Update appropriate self.sampling_bin_prop column from self.hyper_cat_bin_log_probs
            #    and self.sampling_bin_raw_prop column from self.hyper_cat_bin_raw_probs
            self.sampling_bin_prop[:,bin_prop] = np.zeros((len(self.cat_freq))) \
                                                 + self.hyper_cat_bin_log_probs[:,bin_prop]
            self.sampling_bin_raw_prop[:,bin_prop] = np.zeros((len(self.cat_freq))) \
                                                    + self.hyper_cat_bin_raw_probs[:,bin_prop]
            if debug_level >= 9000:
                debug_file.write("After updating self.sampling_bin_prop from new hypers, it looks like this\n")
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.9f')
                debug_file.write(" and self.sampling_bin_raw_prop looks like this\n")
                np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.9f')

            # update appropriate beta
            if bin_prop == 0: #anim_gram_subj
                self.betas['beta0_anim_gram_subj'] = 0 + self.betas['beta0_anim_gram_subj_hyper']
                self.betas['beta1_anim_gram_subj'] = 0 + self.betas['beta1_anim_gram_subj_hyper']
            elif bin_prop == 1: # anim_gram_obj
                self.betas['beta0_anim_gram_obj'] = 0 + self.betas['beta0_anim_gram_obj_hyper']
                self.betas['beta1_anim_gram_obj'] = 0 + self.betas['beta1_anim_gram_obj_hyper']
            elif bin_prop == 2: # anim_gram_iobj
                self.betas['beta0_anim_gram_iobj'] = 0 + self.betas['beta0_anim_gram_iobj_hyper']
                self.betas['beta1_anim_gram_iobj'] = 0 + self.betas['beta1_anim_gram_iobj_hyper']
            elif bin_prop == 3: # mvmt
                self.betas['beta0_mvmt'] = 0 + self.betas['beta0_mvmt_hyper']
                self.betas['beta1_mvmt'] = 0 + self.betas['beta1_mvmt_hyper']

            self.updated_beta_hyper = True    
          
            
    def sample_hyper_gamma(self, debug_level, debug_file):
        self.updated_gamma_hyper = False # set to true if new gamma_c, since need to recalc cat_prob_logs
        # get proposal value
        self.gamma_c_hyper = self.get_proposal_value(debug_level, debug_file, self.gamma_c)
        if debug_level >= 7000:
            debug_file.write("sample_hyper_gamma: current gamma_c = {}, ".format(self.gamma_c))
            debug_file.write("proposed value kappa = {}\n".format(self.gamma_c_hyper))
        # now calculate p(predicates | old value) vs. p(predicates | new value)
        # old value = current log prob for just the categories, but ignoring empty category (so subtract off)
        p_old_hyper = self.calc_hyper_cat_prob(debug_level, debug_file, self.cat_freq)
        if debug_level >= 8000:
            debug_file.write("sample_hyper_gamma: self.cat_freq is currently\n")
            np.savetxt(debug_file, self.cat_freq, fmt='%1.4f')
            debug_file.write(" and p_old_hyper = {}\n".format(p_old_hyper))
        # now do for new value
        # then do same calculation as before    
        p_new_hyper = self.calc_hyper_cat_prob(debug_level, debug_file, self.cat_freq, self.gamma_c_hyper)
        if debug_level >= 8000:
            debug_file.write(" p_new_hyper = {}\n".format(p_new_hyper))
        # now do metropolis hastings sampling with p_old_hyper and p_new_hyper
        # note: need to add something to each one to get rid of underflow issues (pick smaller one)
        p_old_hyper, p_new_hyper = self.get_rid_of_underflow(debug_level, debug_file, p_old_hyper, p_new_hyper)
        
        selected_hyper = self.mh_sampling(debug_level, debug_file, p_old_hyper, p_new_hyper, \
                                     self.gamma_c, self.gamma_c_hyper)
        if debug_level >= 8000:
            debug_file.write("sample_hyper_gamma: selected_hyper = {}\n".format(selected_hyper))
        # if selected_hyper is the new one (= self.gamma_c_hyper), then need to reset things
        # that depend on gamma_c (self.gamma_c, self.cat_freq)
        if selected_hyper: # with index not = 0
            # for non-zero cat_freq entries, subtract old gamma_c and add new gamma_c_hyper
            self.cat_freq[np.nonzero(self.cat_freq)[0]] += self.gamma_c_hyper - self.gamma_c
            self.gamma_c = self.gamma_c_hyper # then update gamma_c
            if debug_level >= 7000:
                debug_file.write("Selected new gamma hyper to be {}\n".format(self.gamma_c))
                debug_file.write(" and self.cat_freq is now\n")
                np.savetxt(debug_file, self.cat_freq, fmt='%1.4f')
            self.updated_gamma_hyper = True

    def get_rid_of_underflow(self, debug_level, debug_file, old_p, new_p):
        to_subtract = max(old_p, new_p)
        if debug_level >= 9000:
            debug_file.write("underflow: subtracting {}\n".format(to_subtract))
            debug_file.write("now, old_p = {}".format(old_p - to_subtract))
            debug_file.write(" and new_p = {}\n".format(new_p - to_subtract))
        return (old_p - to_subtract, new_p - to_subtract)
                                                                                        
    def mh_sampling(self, debug_level, debug_file, old_p, new_p, kappa, kappa_prime, \
                    kappa2=None, kappa2_prime=None): # kappa2 and kappa2_prime are if we're doing the beta hyperparameters
        # a1 = ratio of p with old hyperparameter to p with new hyperparameter        
        a1 = np.exp(old_p)/np.exp(new_p) # Need to get out of log form
        # a2 = p(kappa | normal distrib (kappa_prime, .1*kappa_prime)) /
        #      p(kappa_prime | normal distrib(kappa, .1*kappa))
        a2 = scipy.stats.norm(kappa_prime, .1*kappa_prime).pdf(kappa) / \
             scipy.stats.norm(kappa, .1*kappa).pdf(kappa_prime)
        if kappa2 and kappa2_prime: # need to multiply by probability of drawing kappa2 vs kappa2_prime
            a2 *= scipy.stats.norm(kappa2_prime, .1*kappa2_prime).pdf(kappa2) / \
                  scipy.stats.norm(kappa2, .1*kappa2).pdf(kappa2_prime)
        a = a1*a2
        if debug_level >= 8000:
            debug_file.write("mh_sampling: a1 = {}, a2 = {}, a = {}\n".format(a1, a2, a))
        # if a >= 1, just accept new value kappa_prime
        if a >= 1.0:
            if debug_level >= 8000:
                debug_file.write("mh.sampling: a >= 1, accept new value {}\n".format(kappa_prime))
            return kappa_prime
        else:
            # flip weighted coin to choose which one
            a_array = np.array(([a, 1-a]))
            if debug_level >= 8000:
                debug_file.write("mh_sampling: need to flip coin, a_array = \n")
                np.savetxt(debug_file, a_array, fmt='%1.4f')
            # check if annealing
            if self.with_annealing:
                a_array = np.power(a_array, self.curr_temp) # raise to curr_temp power
                if debug_level >= 8000:
                    debug_file.write("mh_sampling: after annealing, a_array = \n")
                    np.savetxt(debug_file, a_array, fmt='%1.4f')                    
            # get selection
            return(get_rand_weighted(a_array, debug_level, debug_file, False)) # probabilities aren't in log form
                    
    def calc_hyper_cat_prob(self, debug_level, debug_file, cat_freq, new_gamma_c=None):
        # current log prob for just the categories, but ignoring empty category (so subtract off)
        # make sure to ignore 0.0 entries for deleted categories, or else log(0.0) doesn't work
        #  --> nonzeroind = np.nonzero(a)[0]
        nonzero_cat_freq = np.zeros((len(cat_freq[np.nonzero(cat_freq)[0]])))
        nonzero_cat_freq += cat_freq[np.nonzero(cat_freq)[0]]
        if debug_level >= 8000:
            debug_file.write("calc_hyper_cat_prob: nonzero cat freq are\n")
            np.savetxt(debug_file, nonzero_cat_freq, fmt='%1.4f')
        nonzero_cat_freq = np.delete(nonzero_cat_freq, -1)
        if debug_level >= 8000:
            debug_file.write("calc_hyper_cat_prob: nonzero cat freq without empty category are\n")
            np.savetxt(debug_file, nonzero_cat_freq, fmt='%1.4f')
        if new_gamma_c: # doing for proposed hyperparameter
            if debug_level >= 8000:
                debug_file.write("calculating for new_gamma_c {}\n".format(new_gamma_c))
            # now subtract out old gamma_c and add proposed gamma_c
            nonzero_cat_freq -= self.gamma_c
            nonzero_cat_freq += self.gamma_c_hyper
            if debug_level >= 8000:
                debug_file.write("calc_hyper_cat_prob: nonzero cat freq without empty category are\n")
                np.savetxt(debug_file, nonzero_cat_freq, fmt='%1.4f')
        # so summing log_prob for each existing category
        return (np.log(nonzero_cat_freq/nonzero_cat_freq.sum(axis=0)).sum(axis=0))
        
    def get_proposal_value(self, debug_level, debug_file, kappa):
        # sample proposed new value kappa from normal distribution with mean = kappa and sigma = .1*kappa        
        return np.random.normal(kappa, .1*kappa)
        
    def calc_all_cat_prob_logs(self, debug_level, debug_file):
        # use self.category_changed to determine which log_p_binary and log_p_multinomial need to be recaculated
        if debug_level >= 8000:
            debug_file.write("calc_all_cat_prob_logs: self.category_changed looks like this\n")
            np.savetxt(debug_file, self.category_changed, fmt= '%3i')
        # only want to recalculate the entries that have non-zero number of changes
        for cat_index in range(len(self.categories)): 
            # make sure not deleted category (with freq = 0.0) and non-zero number of changes
            if self.cat_freq[cat_index] and self.category_changed[cat_index]:
                if debug_level >= 8000:
                    debug_file.write("category {} has been changed\n".format(cat_index))
                    debug_file.write(" because self.category_changed[{}] = {}\n".\
                                     format(cat_index, self.category_changed[cat_index]))
                # note: category dependent probabilities are already up-to-date since they're
                # recalculated every time a predicate is removed and added
                # it's just the log_p_XXX that need to be recalculated
                if debug_level >= 8000:
                    debug_file.write("Before updating cat prob logs, it is {}\n".format(self.cat_prob_logs[cat_index]))    
                self.calc_cat_prob(debug_level, debug_file, cat_index)        
                if debug_level >= 8000:
                    debug_file.write("After updating cat prob logs, it is {}\n".format(self.cat_prob_logs[cat_index]))  
                    
                
    def get_curr_temp(self, debug_level, debug_file, iter):
        # only need to change if in new increment
        #   with increments of size self.iterations // 20 (ex: 2000 // 20 = increments size 100)
        if iter % (self.iterations // 20) == 0:
            self.curr_temp = self.annealing_schedule[iter // (self.iterations // 20)]
            if debug_level >= 5000:
                debug_file.write("Updating current temp for annealing to {}\n".format(self.curr_temp))            
        
    def sample_predicates(self, debug_level, debug_file):
        if debug_level >= 10000:
            debug_file.write("Sampling predicates\n")
            
        # for each predicate in self.predicates
        for pred in self.predicates:
            if debug_level >= 8000:
                debug_file.write("sample_predicates: sampling category for predicate {}\n".format(pred.pred_name))
            # remove this predicate's info from its current category & update dependent measures
            self.remove_pred_from_cat(debug_level, debug_file, pred.category, pred)

            # sample category, based on self.cat_prob_logs_sampling (and being sensitive to nan entries)
            # when annealing, do (p^T --> to log: T*log(p)) before actually doing sample          
            self.new_cat = self.sample_cat_for_pred(debug_level, debug_file, pred)
            if debug_level >= 8000:
                debug_file.write("new category for pred {} = {}\n".format(pred.pred_name, self.new_cat))
            
            # get new category assignment & add this predicate's info to the new category & update dependent measures
            self.add_pred_after_sampling(debug_level, debug_file, pred)
            # - also want to keep track of how many predicates change category
            #   so if not going back to same category, update log of category changed
            # - if going to same category, self.cat_prob_logs remains the same

    def add_pred_after_sampling(self, debug_level, debug_file, pred):
        # add this predicate's info into self.new_cat & update dependent measures
        self.new_from_zeroed = False # to track if came from just-deleted category and added into "new" category
        cat_index = self.new_cat
        if debug_level >= 9000:
            debug_file.write("New category for pred {} is {}\n".format(pred.pred_name, cat_index))
        # may need to create new category -- check for this
        if (cat_index == len(self.categories)) and not self.zeroed_out: # didn't come from zeroed out category
            debug_file.write("*!*Need to create new category for cat index {}\n".format(cat_index))
            # need to create new category that has this pred in it
            self.categories.append(Category(debug_level, debug_file, \
                                   self.all_frames, pred,\
                                   self.with_UTAH, cat_index))
                                   
            if debug_level >= 8000:
                debug_file.write("After creating new category from pred {}, ".format(pred.pred_name))
                debug_file.write("self.categories has {} members and looks like this:\n".format(len(self.categories)))
                for cat in self.categories:
                    cat.print_cat_info(debug_file, self.with_UTAH)
                
            self.update_with_new_category(debug_level, debug_file, cat_index+1, pred) # add new empty category to end for dependent vars
        else:
            # check if zeroed out category pred came from -- if so, put it back
            if (cat_index == len(self.categories)) and self.zeroed_out:
                self.new_from_zeroed = True
                if debug_level >= 8000:
                    debug_file.write("*^*adding pred back into category it came from, which was zeroed out: {}\n"\
                                     .format(self.prev_cat))
                cat_index = self.prev_cat
                if debug_level >= 9000:
                    debug_file.write("New category for pred {} is now {}\n".format(pred.pred_name, cat_index))
                # re-initialize all the stuff that we zeroed out for this category
                self.cat_freq[cat_index] = self.gamma_c # initialize to gamma_c
                if debug_level >= 9000:
                    debug_file.write("self.cat_freq[{}] is {}\n".format(cat_index, self.cat_freq[cat_index]))
                self.cat_prob_logs[cat_index] = self.old_cat_prob_logs # set the cat_prob_logs entry to old value
                # rely on old vals for log_p_binary and log_p_multinomial
                self.log_p_binary[cat_index] = self.old_cat_log_p_binary
                self.log_p_multinomial[cat_index] = self.old_cat_log_p_multinomial
                if debug_level >= 9000:
                    debug_file.write("After re-initializing category {}, ".format(cat_index))
                    debug_file.write("self.log_p_binary is \n")
                    np.savetxt(debug_file, self.log_p_binary, fmt='%1.4f')
                    debug_file.write("\nself.log_p_multinomial is \n")
                    np.savetxt(debug_file, self.log_p_multinomial, fmt='%1.4f')
                    debug_file.write("\ncategory stats currently look like this:\n")
                    self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)
            else: # do stuff that only applies if using category that wasn't deleted
                # these were left untouched when we deleted the category
                # cat binary properties
                self.categories[cat_index].anim_gram_subj += pred.anim_gram_subj
                self.categories[cat_index].anim_gram_obj += pred.anim_gram_obj
                self.categories[cat_index].anim_gram_iobj += pred.anim_gram_iobj
                if self.with_UTAH:
                    self.categories[cat_index].mvmt += pred.mvmt
                # cat multinomial properties
                self.categories[cat_index].fast_frames += pred.fast_frames
                if not self.with_UTAH:
                    self.categories[cat_index].sem_subj += pred.sem_subj
                    self.categories[cat_index].sem_obj += pred.sem_obj
                    self.categories[cat_index].sem_iobj += pred.sem_iobj
                  
            # if adding to existing category (recently zeroed out or otherwise)
            self.categories[cat_index].num_preds +=1
            # add to category's predicate list
            self.categories[cat_index].predicates.append(pred)

        # for all categories, new or otherwise    
        # add for pred learner: self.cat_freq[cat_index]
        self.cat_freq[cat_index] +=1
        #self.sampling_p_cat = self.cat_freq/self.cat_freq.sum(axis=0) # now sampling_p_cat has been recomputed - don't really need to
        if debug_level >= 9000:
            debug_file.write("After adding & updating self.cat_freq, category looks like this:\n")
            self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)
            debug_file.write("Predicate stats are this:\n")
            pred.print_pred_info(debug_file)
            debug_file.write("self.cat_freq is this\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
            #debug_file.write("self.sampling_p_cat is this\n")
            #np.savetxt(debug_file, self.sampling_p_cat, '%1.4f')            
   

        if not ((cat_index == len(self.categories)) and self.zeroed_out): # otherwise, can use existing binary & multi props    
            # recalculate binary properties for this cat_index -- don't need to update log_p_binary yet, though
            if debug_level >= 9000:
                debug_file.write("After adding & before recalculation, self.sampling_bin_prop is\n");
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.4f')
            self.calc_all_sampling_bin_prop(debug_level, debug_file, cat_index, self.categories[cat_index])
            if debug_level >= 8000:
                debug_file.write("After adding & after recalculation, self.sampling_bin_prop is\n");
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.4f')
            # recalculate multinomial properties for this cat_index -- don't need to update log_p_multinomial yet, though
            if debug_level >= 9000:
                debug_file.write("After adding & before recalculation, self.multinomial props are\n");
                if not self.with_UTAH:
                    debug_file.write("self.sem_subj_probs:\n")
                    np.savetxt(debug_file, self.sem_subj_probs, fmt='%1.4f')
                    debug_file.write("self.sem_obj_probs:\n")                    
                    np.savetxt(debug_file, self.sem_obj_probs, fmt='%1.4f')
                    debug_file.write("self.sem_iobj_probs:\n")                    
                    np.savetxt(debug_file, self.sem_iobj_probs, fmt='%1.4f')
                debug_file.write("self.fast_frames_probs:\n")                    
                np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.4f')  
            self.calc_all_multi_prop(debug_level, debug_file, cat_index)
            if debug_level >= 8000:
                debug_file.write("After adding & after recalculation, self.multinomial props are\n");
                if not self.with_UTAH:
                    debug_file.write("self.sem_subj_probs:\n")
                    np.savetxt(debug_file, self.sem_subj_probs, fmt='%1.4f')
                    debug_file.write("self.sem_obj_probs:\n")                    
                    np.savetxt(debug_file, self.sem_obj_probs, fmt='%1.4f')
                    debug_file.write("self.sem_iobj_probs:\n")
                    np.savetxt(debug_file, self.sem_iobj_probs, fmt='%1.4f')
                debug_file.write("self.fast_frames_probs:\n")                    
                np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.4f')          


        # make note that category cat_index changed and so self.cat_prob_logs will need to be recalculated too
        # Also if not new category when self.zeroed_out -- otherwise, just adding it back in
        if (self.prev_cat != self.new_cat) and not self.new_from_zeroed:
            self.category_changed[cat_index] += 1.0
            pred.category = cat_index # update the predicate's category to cat_index
            debug_file.write("predicate {} category now = {}\n".format(pred.pred_name, pred.category))
        else: # actually put it back in the same category
            if debug_level >= 7000:
                debug_file.write("!!!Went back into same category: {}\n".format(cat_index))
            self.category_changed[cat_index] -= 1.0 # since subtracting it off didn't actually change the category    

        if debug_level >= 9000:
            debug_file.write("After adding, category looks like this:\n")
            self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)
            debug_file.write("Predicate stats are this:\n")
            pred.print_pred_info(debug_file)
            debug_file.write("self.cat_freq is this\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
            debug_file.write("self.sampling_p_cat is this\n")
            np.savetxt(debug_file, self.sampling_p_cat, '%1.4f')
            debug_file.write("self.cat_prob_logs is this\n")
            np.savetxt(debug_file, self.cat_prob_logs, '%1.4f')
            debug_file.write("self.cat_prob_logs_sampling is this\n")
            np.savetxt(debug_file, self.cat_prob_logs_sampling, '%1.4f')
            debug_file.write("self.category_changed is this\n")
            np.savetxt(debug_file, self.category_changed, '%5i') 

    def update_with_new_category(self, debug_level, debug_file, cat_index, pred):
        # create new empty category since old one is being filled
        # update self.cat_freq, self.sampling_p_cat, self.cat_prob_logs, self.cat_prob_log_sampling, 
        #  self.category_changed, log_p_binary, log_p_multinomial, self.sampling_bin_prop,
        #  self.sem_XXX_probs, self.fast_frames_probs
        #   to have new entry at cat_index (should be self.new_cat + 1)
        if debug_level >= 9000:
            debug_file.write("update_with_new_category: self.cat_freq before updating\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
        self.cat_freq = np.append(self.cat_freq, self.gamma_c) # new entry with initial value of gamma_c
        if debug_level >= 9000:
            debug_file.write("update_with_new_category: self.cat_freq after updating\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')           
        self.sampling_p_cat = np.append(self.sampling_p_cat, 0.0) # initialize to 0 -- will be recomputed later
        self.cat_prob_logs = np.append(self.cat_prob_logs, 0.0) # initialize to 0.0 - will recompute when doing hypersampling
        self.cat_prob_logs_sampling = np.append(self.cat_prob_logs_sampling, 0.0) # initialize to 0.0 - will recompute when calc_cat_prob
        self.category_changed = np.append(self.category_changed, 0.0) # since this category has been created, but doesn't matter for log posterior calculations                    
        self.log_p_binary = np.append(self.log_p_binary, 0.0) # initialize to 0.0 -- will recompute when doing hypersampling
        self.log_p_multinomial = np.append(self.log_p_multinomial, 0.0) # initialize to 0.0 -- will recompute when doing hypersampling
        # sampling_bin_raw_prop, sampling_bin_prop: initialize to 0.0
        self.sampling_bin_raw_prop = np.vstack((self.sampling_bin_raw_prop, np.zeros((3+self.with_UTAH), dtype=float)))
        self.sampling_bin_prop = np.vstack((self.sampling_bin_prop, np.zeros((3+self.with_UTAH), dtype=float)))
        #self.calc_all_sampling_bin_prop(debug_level, debug_file, cat_index-1, pred) # set probabilities appropriately using pred's counts -- these will be calculated separately
        self.calc_all_sampling_bin_prop(debug_level, debug_file, cat_index) # and for brand new category
                
        if debug_level >= 9000:
            debug_file.write("self.sampling_bin_prop after adding new zeros row:\n")
            np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.4f')
        # for multinomial properties, need to add whole rows of varying lengths
        if not self.with_UTAH: 
            self.sem_subj_raw_probs = np.vstack((self.sem_subj_raw_probs, (np.zeros((3), dtype=float)+self.alphas['alpha_sem_subj']))) 
            self.sem_obj_raw_probs = np.vstack((self.sem_obj_raw_probs, (np.zeros((3), dtype=float)+self.alphas['alpha_sem_obj'])))
            self.sem_iobj_raw_probs = np.vstack((self.sem_iobj_raw_probs, (np.zeros((3), dtype=float)+self.alphas['alpha_sem_iobj'])))
            self.sem_subj_probs = np.vstack((self.sem_subj_probs, np.zeros((3), dtype=float)))
            self.sem_obj_probs = np.vstack((self.sem_obj_probs, np.zeros((3), dtype=float)))
            self.sem_iobj_probs = np.vstack((self.sem_iobj_probs, np.zeros((3), dtype=float)))
        self.fast_frames_raw_probs = \
          np.vstack((self.fast_frames_raw_probs, (np.zeros((len(self.all_frames)), dtype=float)+self.alphas['alpha_frames']))) # recompute when calc_cat_prob
        self.fast_frames_probs = \
          np.vstack((self.fast_frames_probs, np.zeros((len(self.all_frames)), dtype=float)))
        #self.calc_all_multi_prop(debug_level, debug_file, cat_index-1) # set probabilities appropriately - will be done above
        self.calc_all_multi_prop(debug_level, debug_file, cat_index) # and for brand new category       
        if debug_level >= 7000:
            debug_file.write("After initializing empty category stats, things look like this:\n")
            debug_file.write("self.cat_freq is this\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
            debug_file.write("self.sampling_p_cat is this\n")
            np.savetxt(debug_file, self.sampling_p_cat, '%1.4f')
            debug_file.write("self.cat_prob_logs is this\n")
            np.savetxt(debug_file, self.cat_prob_logs, '%1.4f')
            debug_file.write("self.cat_prob_logs_sampling is this\n")
            np.savetxt(debug_file, self.cat_prob_logs_sampling, '%1.4f')
            debug_file.write("self.category_changed is this\n")
            np.savetxt(debug_file, self.category_changed, '%5i')
            debug_file.write("self.sampling_bin_prop is this\n")
            np.savetxt(debug_file, self.sampling_bin_prop, '%1.4f')
            debug_file.write("self.fast_frames_raw_probs is this\n")
            np.savetxt(debug_file, self.fast_frames_raw_probs, '%1.4f')
            debug_file.write("self.fast_frames_probs is this\n")
            np.savetxt(debug_file, self.fast_frames_probs, '%1.4f')
            if not self.with_UTAH:
                debug_file.write("self.sem_subj_raw_probs is this\n")
                np.savetxt(debug_file, self.sem_subj_raw_probs, '%1.4f')
                debug_file.write("self.sem_subj_probs is this\n")
                np.savetxt(debug_file, self.sem_subj_probs, '%1.4f')
                debug_file.write("self.sem_obj_raw_probs is this\n")
                np.savetxt(debug_file, self.sem_obj_raw_probs, '%1.4f')
                debug_file.write("self.sem_obj_probs is this\n")
                np.savetxt(debug_file, self.sem_obj_probs, '%1.4f')
                debug_file.write("self.sem_iobj_raw_probs is this\n")
                np.savetxt(debug_file, self.sem_iobj_raw_probs, '%1.4f')
                debug_file.write("self.sem_iobj_probs is this\n")
                np.savetxt(debug_file, self.sem_iobj_probs, '%1.4f')  
            
    def sample_cat_for_pred(self, debug_level, debug_file, pred):
        # sample category, based on self.cat_prob_logs_sampling (and being sensitive to nan entries)
        # when annealing, do (p^T --> to log: T*log(p)) before actually doing sample

        # need to transform self.cat_prob_logs_sampling into non-log version
        # first, initialize to_sample_from
        to_sample_from = np.zeros((len(self.cat_prob_logs_sampling)), dtype=float) +\
                         self.cat_prob_logs_sampling # should now be the same as cat_prob_logs_sampling values
        if debug_level >= 8000:
            debug_file.write("sample_cat_for_pred: to_sample_from (before masking) = \n");
            np.savetxt(debug_file, to_sample_from, fmt='%1.4f')
        # create masked array that ignores 0.0 entries    
        masked_to_sample_from = ma.masked_equal(to_sample_from, 0.0)
        if debug_level >= 8000:
            debug_file.write("sample_cat_for_pred: masked_to_sample_from = \n");
            np.savetxt(debug_file, masked_to_sample_from, fmt='%1.4f')
        if self.with_annealing:
            # multiple all values by self.curr_temp
            masked_to_sample_from *= self.curr_temp
            if debug_level >= 8000:
                debug_file.write("sample_cat_for_pred: after annealing, masked_to_sample_from = \n");
                np.savetxt(debug_file, masked_to_sample_from, fmt='%1.4f')     
        # send to get_rand_weighted to get index back
        return get_rand_weighted(masked_to_sample_from, debug_level, debug_file, True)

                
    def remove_pred_from_cat(self, debug_level, debug_file, cat_index, pred):
        # remove this predicate's info from its current category & update dependent measures
        self.zeroed_out = False # to track if this was the last pred in this category
        # pred.category = category predicate currently belongs to
        self.prev_cat = pred.category # keep track of this in case it doesn't end up changing - compare against self.new_cat
        if debug_level >= 9000:
            debug_file.write("Previous category = {}\n".format(pred.category))

        # subtract stats from self.categories[cat_index]
        if debug_level >= 9000:
            debug_file.write("Subtracting predicate {} stats from category {}\n".format(\
                              pred.pred_name, cat_index))
            debug_file.write("Before subtracting, category looks like this:\n")
            self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)
            debug_file.write("Predicate stats are this:\n")
            pred.print_pred_info(debug_file)
            debug_file.write("self.cat_freq is this\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
        # subtract off for this category: freq, place in pred list, binary vars, multinomial vars
        self.categories[cat_index].num_preds -= 1
        # remove from category's predicate list
        self.categories[cat_index].predicates.remove(pred)  
        if(self.categories[cat_index].num_preds == 0): # if no predicates now left in this category
            if debug_level >= 5000:
                debug_file.write("***No preds left in category {} after removing {} -- will do clean up\n".format(\
                                 cat_index, pred.pred_name))
            self.zeroed_out = True                     
            # do category clean-up if this is the only predicate in the category
            self.cat_freq[cat_index] = 0.0 # now this category's frequency is 0.0
            #self.sampling_p_cat = self.cat_freq/(self.cat_freq.sum(axis=0) - 1.0) # now sampling_p_cat has been recomputed (again)
            # save old cat_prob_logs
            self.old_cat_prob_logs = self.cat_prob_logs[cat_index]
            self.cat_prob_logs[cat_index] = np.nan # set the cat_prob_logs entry to nan
            #self.cat_prob_logs_sampling[cat_index] = 0.0 # set the cat_prob_logs_sampling entry to 0.0 so can skip when doing sampling
            # save old log_p_binary and log_p_multinomial vals in case goes back in same category
            self.old_cat_log_p_binary = self.log_p_binary[cat_index]
            self.old_cat_log_p_multinomial = self.log_p_multinomial[cat_index]
            self.log_p_binary[cat_index] = np.nan # log prob for binary properties now is nan -- probably not necessary
            self.log_p_multinomial[cat_index] = np.nan # log prob for multinomial properties now is nan -- probably not necessary
            # These will then effectively be skipped over when it comes to summing and sampling 
            # don't need to remove from self.categories, since that would mess up the indexing
        else:            
            # cat binary properties
            self.categories[cat_index].anim_gram_subj -= pred.anim_gram_subj
            self.categories[cat_index].anim_gram_obj -= pred.anim_gram_obj
            self.categories[cat_index].anim_gram_iobj -= pred.anim_gram_iobj
            if self.with_UTAH:
                self.categories[cat_index].mvmt -= pred.mvmt
            # cat multinomial properties
            self.categories[cat_index].fast_frames -= pred.fast_frames
            if not self.with_UTAH:
                self.categories[cat_index].sem_subj -= pred.sem_subj
                self.categories[cat_index].sem_obj -= pred.sem_obj
                self.categories[cat_index].sem_iobj -= pred.sem_iobj
          
            # subtract off for pred learner: self.cat_freq[cat_index]
            self.cat_freq[cat_index] -=1
            self.sampling_p_cat = self.cat_freq/(self.cat_freq.sum(axis=0) - 1.0) # now sampling_p_cat has been recomputed  
             
            # recalculate binary properties for this cat_index -- don't need to update log_p_binary yet, though
            if debug_level >= 9000:
                debug_file.write("After subtracting & before recalculation, self.sampling_bin_raw_prop is\n");
                np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.4f')
            self.calc_all_sampling_bin_prop(debug_level, debug_file, cat_index, self.categories[cat_index])
            if debug_level >= 8000:
                debug_file.write("After subtracting & after recalculation, self.sampling_bin_raw_prop is\n");
                np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.4f')
            # recalculate multinomial properties for this cat_index -- don't need to update log_p_multinomial yet, though
            if debug_level >= 9000:
                debug_file.write("After subtracting & before recalculation, self.multinomial props are\n");
                if not self.with_UTAH:
                    debug_file.write("self.sem_subj_raw_probs:\n")
                    np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.4f')
                    debug_file.write("self.sem_obj_raw_probs:\n")                    
                    np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.4f')
                    debug_file.write("self.sem_iobj_raw_probs:\n")                    
                    np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.4f')
                debug_file.write("self.fast_frames_raw_probs:\n")                    
                np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.4f')    
            self.calc_all_multi_prop(debug_level, debug_file, cat_index)
            if debug_level >= 8000:
                debug_file.write("After subtracting & after recalculation, self.multinomial props are\n");
                if not self.with_UTAH:
                    debug_file.write("self.sem_subj_raw_probs:\n")
                    np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.4f')
                    debug_file.write("self.sem_obj_raw_probs:\n")                    
                    np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.4f')
                    debug_file.write("self.sem_iobj_raw_probs:\n")                    
                    np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.4f')
                debug_file.write("self.fast_frames_raw_probs:\n")                    
                np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.4f')

        # calculate cat_log_probs_sampling for this predicate - do these all at once since they change all the time
        self.calc_all_cat_log_probs_sampling(debug_level, debug_file, pred)
                    
        # make note that category cat_index changed and so self.cat_prob_logs will need to be recalculated too
        # to get correct log posterior
        self.category_changed[cat_index] += 1.0

            
        if debug_level >= 8000:
            debug_file.write("\nAfter subtracting, category looks like this:\n")
            self.categories[cat_index].print_cat_info(debug_file, self.with_UTAH)
            debug_file.write("Predicate stats are this:\n")
            pred.print_pred_info(debug_file)
            debug_file.write("self.cat_freq is this\n")
            np.savetxt(debug_file, self.cat_freq, '%1.4f')
            #debug_file.write("self.sampling_p_cat is this\n")
            #np.savetxt(debug_file, self.sampling_p_cat, '%1.4f')
            #debug_file.write("self.cat_prob_logs is this\n")
            #np.savetxt(debug_file, self.cat_prob_logs, '%1.4f')
            #debug_file.write("self.cat_prob_logs_sampling is this\n")
            #np.savetxt(debug_file, self.cat_prob_logs_sampling, '%1.4f')
            debug_file.write("self.category_changed is this\n")
            np.savetxt(debug_file, self.category_changed, '%5i') 
                
    def calc_all_cat_log_probs_sampling(self, debug_level, debug_file, pred):
        ## log cat prob = log(sampling_p_cat[cat_index])
        self.sampling_p_cat = self.cat_freq/self.cat_freq.sum(axis=0)
        if debug_level >= 9000:
            debug_file.write("calc_all_cat_log_probs_sampling: self.cat_freq is\n")
            np.savetxt(debug_file, self.cat_freq, fmt='%1.4f')
            debug_file.write("calc_all_cat_log_probs_sampling: self.sampling_p_cat is\n")
            np.savetxt(debug_file, self.sampling_p_cat, fmt='%1.4f')
        # want to ignore entries that have 0.0
        log_sampling_p_cat = np.zeros((len(self.sampling_p_cat))) + self.sampling_p_cat
        for cat_index in np.nonzero(self.sampling_p_cat)[0]: 
            log_sampling_p_cat[cat_index] = log(self.sampling_p_cat[cat_index])
                
        if debug_level >= 9000:
            debug_file.write("log of those probs is\n")        
            np.savetxt(debug_file, log_sampling_p_cat, fmt='%1.4f')            
        
        if debug_level >= 9000:
            debug_file.write("calc_all_cat_log_probs_sampling: binary props for pred {}\n".format(pred.pred_name))
            debug_file.write("before calculating for this pred, self.sampling_bin_raw_prop is\n")
            np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.4f')
            debug_file.write("\nand pred anim_gram_subj is\n")
            np.savetxt(debug_file, pred.anim_gram_subj, fmt='%1.4f')
            debug_file.write("\nand pred anim_gram_obj is\n")
            np.savetxt(debug_file, pred.anim_gram_obj, fmt='%1.4f')
            debug_file.write("\nand pred anim_gram_iobj is\n")
            np.savetxt(debug_file, pred.anim_gram_iobj, fmt='%1.4f')
            if self.with_UTAH:
                debug_file.write("\nand pred mvmt is\n")
                np.savetxt(debug_file, pred.mvmt, fmt='%1.4f')

        ## basic form for bin: (n_yes_bin * log(p_bin_prop)) + ((n_no_bin)*log(1-p_bin_prop))
        #                      n_yes_bin = pred.anim_gram_XXX[1],
        #                      n_all_bin = pred.anim_gram_XXX.sum(axis=0)
        #                      p_bin_prop = self.sampling_bin_raw_prop[0 or 1 or 2 or 3]
        # --> go by columns [0 = anim_gram_subj, etc.]
        log_sampling_p_bin = np.zeros((len(self.sampling_p_cat), 3 + self.with_UTAH))
        if debug_level >= 9000:
            debug_file.write("initially log_sampling_p_bin is \n")
            np.savetxt(debug_file, log_sampling_p_bin, fmt='%1.4f')
        # do for anim_gram_subj
        #p_bin_prop = self.sampling_bin_raw_prop[:,0]
        #if debug_level >= 9000:
        #    debug_file.write("anim_gram_subj, p_bin_prop =\n")
        #    np.savetxt(debug_file, p_bin_prop, fmt='%1.4f')
        
        # Make this so only operates over non-zero self.cat_freq entries
        for cat_index in np.nonzero(self.cat_freq)[0]:
            # do for anim_gram_subj
            log_sampling_p_bin[cat_index,0] += pred.anim_gram_subj[1] * np.log(self.sampling_bin_raw_prop[cat_index,0]) +\
                                              (pred.anim_gram_subj[0]) * (np.log(1-self.sampling_bin_raw_prop[cat_index,0]))
            # do for anim_gram_obj
            log_sampling_p_bin[cat_index,1] += pred.anim_gram_obj[1] * np.log(self.sampling_bin_raw_prop[cat_index,1]) +\
                                              (pred.anim_gram_obj[0]) * (np.log(1-self.sampling_bin_raw_prop[cat_index,1]))
            # do for anim_gram_iobj
            log_sampling_p_bin[cat_index,2] += pred.anim_gram_iobj[1] * np.log(self.sampling_bin_raw_prop[cat_index,2]) +\
                                              (pred.anim_gram_iobj[0]) * (np.log(1-self.sampling_bin_raw_prop[cat_index,2]))
            if self.with_UTAH: # do for mvmt
                log_sampling_p_bin[cat_index,3] += pred.mvmt[1] * np.log(self.sampling_bin_raw_prop[cat_index,3]) +\
                                                  (pred.mvmt[0]) * (np.log(1-self.sampling_bin_raw_prop[cat_index,3]))
        if debug_level >= 9000:
            debug_file.write("\nafter bin calculations, log_sampling_p_bin is \n")
            np.savetxt(debug_file, log_sampling_p_bin, fmt='%1.4f')  

        # multinomial properties
        ## basic form for multi: (pred.sem_subj * np.log(self.sem_subj_raw_probs[cat_index])).sum(axis=0)
        if debug_level >= 9000:
            debug_file.write("calc_all_cat_log_probs_sampling: multi props for pred {}\n".format(pred.pred_name))
            if not self.with_UTAH:
                debug_file.write("before calculating for this pred, self.sem_subj_raw_probs is\n")
                np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.4f')
                debug_file.write("self.sem_obj_raw_probs is\n")
                np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.4f')
                debug_file.write("self.sem_iobj_raw_probs is\n")
                np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.4f')                
            debug_file.write("before calculating for this pred, self.fast_frames_raw_probs is\n")
            np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.4f')

        if not self.with_UTAH:
            log_sampling_p_sem_subj = np.zeros((len(self.cat_freq), 3))
            log_sampling_p_sem_obj = np.zeros((len(self.cat_freq), 3))
            log_sampling_p_sem_iobj = np.zeros((len(self.cat_freq), 3))
        log_sampling_p_fast_frames = np.zeros((len(self.cat_freq), len(self.all_frames)))

        if debug_level >= 9000:
            if not self.with_UTAH:
                debug_file.write("before calculating sem_subj,  log_sampling_p_sem_subj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_subj, fmt='%1.4f')
                debug_file.write("before calculating sem_obj,  log_sampling_p_sem_obj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_obj, fmt='%1.4f')
                debug_file.write("before calculating sem_iobj,  log_sampling_p_sem_iobj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_iobj, fmt='%1.4f')
            debug_file.write("before calculating fast_frames,  log_sampling_p_fast_frames is\n")
            np.savetxt(debug_file, log_sampling_p_fast_frames, fmt='%1.4f')
            
        # Make this so only operates over non-zero self.cat_freq entries
        for cat_index in np.nonzero(self.cat_freq)[0]: 
            if not self.with_UTAH:
                if debug_level >= 9500:
                    debug_file.write("pred.sem_subj counts are\n")
                    np.savetxt(debug_file, pred.sem_subj, fmt='%1.4f')   
                log_sampling_p_sem_subj[cat_index] = (pred.sem_subj * np.log(self.sem_subj_raw_probs[cat_index])).sum(axis=0)
                #if debug_level >= 9000:
                #    debug_file.write("after calculating sem_subj, before summing over cat, log_sampling_p_sem_subj is\n")
                #    np.savetxt(debug_file, log_sampling_p_sem_subj, fmt='%1.4f')
                #log_sampling_p_sem_subj = log_sampling_p_sem_subj.sum(axis=1) 
                log_sampling_p_sem_obj[cat_index] = (pred.sem_obj * np.log(self.sem_obj_raw_probs[cat_index])).sum(axis=0)
                log_sampling_p_sem_iobj[cat_index] = (pred.sem_iobj * np.log(self.sem_iobj_raw_probs[cat_index])).sum(axis=0)
  
            log_sampling_p_fast_frames[cat_index] = (pred.fast_frames * np.log(self.fast_frames_raw_probs[cat_index])).sum(axis=0)
                
        if debug_level >= 9000:
            if not self.with_UTAH:
                debug_file.write("after calculating sem_subj, log_sampling_p_sem_subj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_subj, fmt='%1.4f')
                debug_file.write("after calculating sem_obj,  log_sampling_p_sem_obj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_obj, fmt='%1.4f')
                debug_file.write("after calculating sem_iobj,  log_sampling_p_sem_iobj is\n")
                np.savetxt(debug_file, log_sampling_p_sem_iobj, fmt='%1.4f')
            debug_file.write("after calculating fast_frames, log_sampling_p_fast_frames is\n")
            np.savetxt(debug_file, log_sampling_p_fast_frames, fmt='%1.4f')

        # need to do this so only for non-zero entries: use np.nonzero(self.cat_freq)[0]
        self.cat_prob_logs_sampling = np.zeros((len(self.cat_freq)))
        for cat_index in np.nonzero(self.cat_freq)[0]:
            self.cat_prob_logs_sampling[cat_index] = log_sampling_p_cat[cat_index] \
                                                    + log_sampling_p_bin[cat_index].sum() \
                                                    + log_sampling_p_fast_frames[cat_index].sum()
            if not self.with_UTAH:
                self.cat_prob_logs_sampling[cat_index] += log_sampling_p_sem_subj[cat_index].sum() \
                                                          + log_sampling_p_sem_obj[cat_index].sum() \
                                                          + log_sampling_p_sem_iobj[cat_index].sum()
                                           
        #if debug_level >= 9000:
        #    debug_file.write("Before zeroing out, self.cat_prob_logs_sampling is \n")
        #    np.savetxt(debug_file, self.cat_prob_logs_sampling, fmt='%1.4f')
                                           
        # zero out any that are 0.0 in self.cat_freq
        #self.cat_prob_logs_sampling *= (self.cat_freq > 0.0)
        
        if debug_level >= 9000:
            debug_file.write("After all cat calculations, self.cat_prob_logs_sampling is \n")
            np.savetxt(debug_file, self.cat_prob_logs_sampling, fmt='%1.4f')
                                        
    def sample_init(self, debug_level, debug_file, output_file):
        # need to know how many predicate types total (n_k) - this doesn't change so just calculate once
        self.total_pred = len(self.predicates) 
        print "Initializing sampling for {} predicates...".format(self.total_pred)
        output_file.write("\nAfter reading in data, total predicates =  {} ".format(self.total_pred))
        output_file.write("and total categories = {}\n".format(len(self.categories)))

        # sampling pre-computations
        # Begin by pre-calculating each category's probabilities
        # (labels, binary properties, multinomial properties)
        # Then, when sampling, only need to re-compute the values for
        #    (1) the category that the predicate is from since that one had its stats changed by the subtraction
        #    (2) the category that the predicate goes to since that one will have its stats changes by the addition    
        # for each existing category cat_j & new category c_new
 
        #   calculate probability of this label (p_label)
        #   - can do this by using self.cat_freq, which contains counts of predicates in categories
        #   - p_cat_j = (n_j + gamma_c)/(n_all + C*gamma_c)
        #             = self.cat_freq [includes gamma_c] / self.cat_freq.sum(axis=0) [includes all gamma_c] - 1
        #            --> denominator includes possibility of creating new category
        #            --> subtracting 1 in denominator for whichever predicate is removed
        #                because will have to recalculate that category's p_cat anyway, but rest
        #                can remain the same while that predicate is being sampled
        #n_all = self.cat_freq.sum(axis=0)
        self.sampling_p_cat = self.cat_freq/self.cat_freq.sum(axis=0) #(n_all - 1.0)
        if debug_level >= 7000:
            debug_file.write("\nAfter initializing self.sampling_p_cat, ")
            debug_file.write("self.cat_freq is \n")
            print_array(self.cat_freq, debug_file)
            debug_file.write("\nand self.sampling_p_cat is \n")
            print_array(self.sampling_p_cat, debug_file)
            debug_file.write("\n") 
        
        #   for each binary property, calculate probability of -bin-prop and +bin-prop
        #   - 2D array: category x +binary property (note: p(-bin_prop) = 1 - p(+bin_prop))
        self.init_bin_prop_probs(debug_level, debug_file)
        
        #   for each multinomial property, calculate probability of all options
        #   - 2D array for each multinomial property (sem_XXX_gram_YYY [if no UTAH], fast_frames)
        #   - each one: Category x multinomial options
        self.init_multi_prop_probs(debug_level, debug_file)
        
        #   total cat probability = p_label * binary property probs * multinomial property probs
        #   ---> Note: need to do p_cat calculation without - 1.0 in denom for getting this
        #              outside of sampling (i.e., log posterior eval and hypersam eval below)
        self.init_cat_prob(debug_level, debug_file)

        # Also, need to set current temperature at beginning of initialization, since will be getting at beginning
        # of initial iteration run
        if self.with_annealing:
            self.curr_temp = self.annealing_schedule[0]
            if debug_level >= 7000:
                debug_file.write("Initial temperature = {}\n".format(self.curr_temp))
                                
        # hypersampling pre-computations
        #    (1) p_cat calculation doesn't include subtracting 1 off denominator since this
        #        calculation will always include all current predicates
        #     ...but this already handled in the init_cat_prob routine, which calls
        #        calc_cat_prob with the default "False" for sampling
        #        so the p_label calculation is done without subtracting 1.0
        #     All other calculations are the same as for sampling (p_label, p_binary_props, p_multi_props)

        # print out initial log posterior 
        # TO DO: Make log_posterior array for holding log_posterior values over iterations?
        if debug_file >= 1000:
            debug_file.write("Initial log posterior = {}\n".format(self.log_posterior()))
        output_file.write("Initial log posterior = {}\n".format(self.log_posterior()))

        # initialize self.category_changed as array of booleans -- used for recalculating self.cat_prob_logs
        # after iteration completed and before hyperparameter sampling
        self.category_changed = np.zeros((len(self.cat_prob_logs)), dtype=float) # initialized to 0.0
        if debug_level >= 8000:
            debug_file.write("self.category_changed initialized:\n")
            np.savetxt(debug_file, self.category_changed, fmt='%5i')
                                         

    def log_posterior(self):
        # use self.cat_prob_logs, since log posterior = log (category hypotheses)
        return np.nansum(self.cat_prob_logs) # ignore nan entries
        
    def init_cat_prob(self, debug_level, debug_file):
        # initialize self.cat_prob_logs to all zeros, including an entry for the new possible category
        self.cat_prob_logs = zeros((len(self.sampling_p_cat))) # during initialization, have to calculate them all (for logp)
        self.cat_prob_logs_sampling = zeros((len(self.sampling_p_cat))) # but also want to pre-fill the sampling probs
        self.log_p_binary = zeros((len(self.sampling_p_cat))) # save for future calculations
        self.log_p_multinomial = zeros((len(self.sampling_p_cat))) # save for future calculations 
        # (during sampling, have to calculate all p_labels, but only binary & multinomial property probs that have changed
        for cat_index in range(len(self.sampling_p_cat)):
            # total cat probability = p_label * binary property probs * multinomial property probs
            # Note: Doing log, so = log(p_label) + log(binary props) + log(multinomial props)
            self.calc_cat_prob(debug_level, debug_file, cat_index)

        # now initialize log_p_binary and log_p_multinomial for each pred -- no longer necessary
        #for pred in self.predicates:
        #    pred.log_p_multinomial = self.calc_log_p_multinomial(debug_level, debug_file, pred=pred)
        #    pred.log_p_binary = self.calc_log_p_binary(debug_level, debug_file, pred=pred)
        if debug_level >= 7000:
            debug_file.write("After initialization, self.cat_prob_logs is\n")
            np.savetxt(debug_file, self.cat_prob_logs, fmt = '%1.9f')        
            debug_file.write("and self.cat_prob_logs_sampling is\n")
            np.savetxt(debug_file, self.cat_prob_logs_sampling, fmt = '%1.9f') 
                            
       
    def calc_cat_prob(self, debug_level, debug_file, cat_index, sampling=False, pred=None):
        # total cat prob = p_label * binary property probs * multinomial property probs
        # note: probably want to log these and add them since the numbers will get small 
        #       log(p_1^F1 * p_2^F2) = F1*log(p_1) + F2*log(p_2)
        #       then unlog (exp) final cat_prob before doing weighted sampling
        log_p_cat = log(self.calc_p_label(debug_level, debug_file, cat_index, sampling))
        if debug_level >= 8000:
            debug_file.write("calc_cat_prob: Sampling? {}, log(p_cat) for {} = {}\n".format(sampling, cat_index, log_p_cat))
        if not pred: # doing for entire category    
            log_p_binary = self.calc_log_p_binary(debug_level, debug_file, cat_index, sampling)
            if not sampling:
                self.log_p_binary[cat_index] = log_p_binary # save for log posterior calculation
            if debug_level >= 8000:
                debug_file.write("calc_cat_prob: Sampling? {}, log(p_binary) for {} = {}\n".format(sampling, cat_index, log_p_binary))
            log_p_multinomial = self.calc_log_p_multinomial(debug_level, debug_file, cat_index, sampling)
            if not sampling: 
                self.log_p_multinomial[cat_index] = log_p_multinomial # save for log posterior calculation
            if debug_level >= 8000:
                debug_file.write("calc_cat_prob: Sampling? {}, log(p_multinomial) for {} = {}\n".format(sampling, cat_index, log_p_multinomial))
        else: # during sampling, and doing for predicate probability of coming from category cat_index        
            log_p_binary = self.calc_log_p_binary(debug_level, debug_file, cat_index, sampling, pred) 
            if debug_level >= 8000:
                debug_file.write("calc_cat_prob, for pred: Sampling? {}, log(p_binary) for {} = {}\n".format(sampling, cat_index, log_p_binary))
            log_p_multinomial = self.calc_log_p_multinomial(debug_level, debug_file, cat_index, sampling, pred)
            if debug_level >= 8000:
                debug_file.write("calc_cat_prob, for pred: Sampling? {}, log(p_multinomial) for {} = {}\n".format(sampling, cat_index, log_p_multinomial))
                
        if not sampling:
            self.cat_prob_logs[cat_index] = log_p_cat + log_p_binary + log_p_multinomial
        else: # sampling, and so should update with pred posterior predictive
            self.cat_prob_logs_sampling[cat_index] = log_p_cat + log_p_binary + log_p_multinomial    

    def calc_log_p_multinomial(self, debug_level, debug_file, cat_index=None, sampling=False, pred=None):
        log_p_multinomial = log(1) # initialize (will be 0)
        if not pred: # doing for category with cat_index
            if not self.with_UTAH: # do sem_XXX
                log_p_multinomial += self.sem_subj_probs[cat_index].sum(axis=0)
                if debug_level >= 9000:
                    debug_file.write("calc_p_multinomial: After adding sum of self.sem_subj_probs[cat_index], which is\n")
                    np.savetxt(debug_file, self.sem_subj_probs[cat_index], fmt = '%1.9f')
                    debug_file.write("\n(and its sum is {})\n".format(self.sem_subj_probs[cat_index].sum(axis=0)))
                if debug_level >= 7000:    
                    debug_file.write("\nlog_p_multinomial is now {} after adding sem_subj\n".format(log_p_multinomial))
                log_p_multinomial += self.sem_obj_probs[cat_index].sum(axis=0)
                if debug_level >= 7000:    
                    debug_file.write("\nlog_p_multinomial is now {} after adding sem_obj\n".format(log_p_multinomial))
                log_p_multinomial += self.sem_iobj_probs[cat_index].sum(axis=0)
                if debug_level >= 7000:    
                    debug_file.write("\nlog_p_multinomial is now {} after adding sem_iobj\n".format(log_p_multinomial))
            # do fast_frames no matter what
            if debug_level >= 9000:
                debug_file.write("Before adding self.fast_frames_probs[{}] sum to log_p_multinomial,".format(cat_index))
                debug_file.write(" self.fast_frames_probs looks like this:\n")
                np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.4f')
                debug_file.write(" and category {}'s fast_frames are this row\n".format(cat_index))
                np.savetxt(debug_file, self.fast_frames_probs[cat_index], fmt='%1.4f')
            log_p_multinomial += self.fast_frames_probs[cat_index].sum(axis=0)
            if debug_level >= 7000:    
                debug_file.write("\nlog_p_multinomial is now {} after adding fast_frames\n".format(log_p_multinomial))
        else: # doing for individual predicate
            if not sampling: # during initialization
                if not self.with_UTAH:
                    log_p_multinomial += pred.sem_subj_probs.sum(axis=0)
                    log_p_multinomial += pred.sem_obj_probs.sum(axis=0)
                    log_p_multinomial += pred.sem_iobj_probs.sum(axis=0)
                # do fast_frames no matter what
                log_p_multinomial += pred.fast_frames_probs.sum(axis=0)
            else: # during sampling -- calculate using category cat_index's sem_XXX raw probs and pred's counts
                if debug_level >= 9000:
                    debug_file.write("calc_log_p_multinomial, during sampling, doing for pred {} ".format(pred.pred_name))
                    debug_file.write("with category {} probabilities\n".format(cat_index))
                # basic form: sem_XXX_probs = (pred.sem_XXX_count) * np.log(self.categories[cat_index].sem_XXX_raw_probs)
                log_p_multinomial = 0.0   
                if not self.with_UTAH: # sem_XXX
                    if debug_level >= 9000:
                        debug_file.write("calc_log_p_multinomial, not self.with_UTAH: pred {} pred.sem_subj counts = \n".format(\
                                        pred.pred_name))
                        np.savetxt(debug_file, pred.sem_subj, fmt='%1.4f')
                        debug_file.write("category {} sem_subj_raw_probs = \n".format(\
                                        cat_index))
                        np.savetxt(debug_file, self.sem_subj_raw_probs[cat_index], fmt='%1.4f')
                        debug_file.write("log p sem subj sum = {}\n".format((pred.sem_subj * np.log(self.sem_subj_raw_probs[cat_index])).sum(axis=0)))
                        
                    log_p_multinomial += (pred.sem_subj * np.log(self.sem_subj_raw_probs[cat_index])).sum(axis=0) +\
                                        (pred.sem_obj * np.log(self.sem_obj_raw_probs[cat_index])).sum(axis=0) +\
                                        (pred.sem_iobj * np.log(self.sem_iobj_raw_probs[cat_index])).sum(axis=0) 
                # do fast_frames no matter what
                log_p_multinomial += (pred.fast_frames * np.log(self.fast_frames_raw_probs[cat_index])).sum(axis=0)                                                      
                                
        return log_p_multinomial
                
    def calc_log_p_binary(self, debug_level, debug_file, cat_index=None, sampling=False, pred=None):
        log_p_binary = log(1) # initialize (will be 0)
        if not pred: # doing for category with cat_index
            log_p_binary += self.sampling_bin_prop.sum(axis=1)[cat_index]
            if debug_level >= 8000:
                debug_file.write("calc_log_p_binary for category {}, sampling_bin_prop is \n".format(cat_index))
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.4f')
                debug_file.write("calc_log_p_binary: after summing across anim_XXX row for cat {}, ".format(cat_index))
                debug_file.write("log_p_binary is {}\n".format(log_p_binary))
        else: # doing for pred - initial or during sampling?
            if not sampling:
                if debug_level >= 9000:
                    debug_file.write("calc_log_p_binary, not during sampling for predicate {}\n".format(pred.pred_name))
                    debug_file.write("pred.sampling_bin_prop is \n")
                    np.savetxt(debug_file, pred.sampling_bin_prop, fmt='%1.4f')
                    debug_file.write("\nand log_b_binary is {}\n".format(pred.sampling_bin_prop.sum(axis=0)))                         
                log_p_binary += pred.sampling_bin_prop.sum(axis=0)
                
            else: # during sampling -- calculate using category cat_index's sampling_bin_prop
                # basic form: (n_yes_bin * log(p_bin_prop)) + ((n_all_bin - n_yes_bin)*log(1-p_bin_prop)
                # n_yes_bin = pred.anim_gram_subj[1]
                # n_all_bin = pred.anim_gram_subj.sum(axis=0)
                # p_bin_prop = self.sampling_bin_raw_prop[cat_index, XXX]
                if debug_level >= 9000:
                    debug_file.write("calc_log_p_binary, during sampling, doing for pred {} ".format(pred.pred_name))
                    debug_file.write("with category {} probabilities\n".format(cat_index))  
                log_p_binary = self.calc_pred_log_p_binary(debug_level, debug_file, pred.anim_gram_subj[1],\
                                                           pred.anim_gram_subj.sum(axis=0), \
                                                           self.sampling_bin_raw_prop[cat_index, 0]) +\
                               self.calc_pred_log_p_binary(debug_level, debug_file, pred.anim_gram_obj[1],\
                                                           pred.anim_gram_obj.sum(axis=0), \
                                                           self.sampling_bin_raw_prop[cat_index, 1]) +\
                               self.calc_pred_log_p_binary(debug_level, debug_file, pred.anim_gram_iobj[1],\
                                                           pred.anim_gram_iobj.sum(axis=0), \
                                                           self.sampling_bin_raw_prop[cat_index, 2])
                if self.with_UTAH:
                    log_p_binary += self.calc_pred_log_p_binary(debug_level, debug_file, pred.mvmt[1],\
                                                           pred.mvmt.sum(axis=0), \
                                                           self.sampling_bin_raw_prop[cat_index, 3])
                                                           
        return log_p_binary

    def calc_pred_log_p_binary(self, debug_level, debug_file, n_yes_bin, n_all_bin, p_bin_prop):
        if debug_level >= 9000:
            debug_file.write("calc_pred_log_p_binary: n_yes_bin = {}, n_all_bin = {}, p_bin_prop = {}\n".format(\
                             n_yes_bin, n_all_bin, p_bin_prop))
        return (n_yes_bin * log(p_bin_prop)) + ((n_all_bin - n_yes_bin)*log(1-p_bin_prop))
    
    def calc_p_label(self, debug_level, debug_file, cat_index, sampling):       
        #  if sampling: self.sampling_p_cat[cat_index] (since already removed one predicate)
        #  if not sampling: self.cat_freq[cat_index]/self.cat_freq.sum(axis=0) (since not removing any predicates)
        if sampling:
            return self.sampling_p_cat[cat_index]
        else:
            if debug_level >= 9000:
                debug_file.write("calculating {}/{}\n".format(self.cat_freq[cat_index],self.cat_freq.sum(axis=0))) 
            return (self.cat_freq[cat_index]/self.cat_freq.sum(axis=0))

    def init_multi_prop_probs(self, debug_level, debug_file):
        #- 2D array for each multinomial property (sem_XXX_gram_YYY [if no UTAH], fast_frames)
        #   - each one: Category x multinomial options
        if(not self.with_UTAH):
            # initialize one 2D array for each sem_XXX option (sem_subj, sem_obj, sem_iobj)
            self.init_sem_prop_probs(debug_level, debug_file)
        # then do frames (with fast_frames)
        self.init_frame_probs(debug_level, debug_file)

    def calc_all_multi_prop(self, debug_level, debug_file, cat_index):
        if not self.with_UTAH:
            # update array entry for each sem_XXX option (sem_subj, sem_obj, sem_iobj)
            self.calc_sem_probs_cat(debug_level, debug_file, cat_index)
        # then update array entry for fast_frames
        if debug_level >= 9000:
            debug_file.write("Recalculating self.fast_frames_probs[{}]\n".format(cat_index))
            debug_file.write("Before recalculating, self.fast_frames_probs is\n")
            np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.4f')
            if cat_index != len(self.categories):
                debug_file.write("self.categories[{}].fast_frames is:\n".format(cat_index))
                np.savetxt(debug_file, self.categories[cat_index].fast_frames, fmt = '%1.4f')
            debug_file.write("self.alphas['alpha_frames'] = {}\n".format(self.alphas['alpha_frames']))
        self.calc_frames_probs_cat(debug_level, debug_file, cat_index)         

    def init_frame_probs(self, debug_level, debug_file):
        # 2D array: Category x # of frames in self.all_frames
        self.fast_frames_raw_probs = np.zeros((len(self.sampling_p_cat), len(self.all_frames)))     # initialize to zeros
        self.fast_frames_probs = np.zeros((len(self.sampling_p_cat), len(self.all_frames)))     # initialize to zeros, holds log probs
        for cat_index in range(len(self.categories)+1): # also do for empty category
           self.calc_frames_probs_cat(debug_level, debug_file, cat_index)  
        if debug_level >= 8000:
            debug_file.write("self.fast_frames_raw_probs totals before normalization:\n")
            np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.4f')
        for pred in self.predicates:
            pred.fast_frames_raw_probs = np.zeros((len(pred.fast_frames)))
            pred.fast_frames_probs = np.zeros((len(pred.fast_frames)))
            self.calc_frames_probs_cat(debug_level, debug_file, pred=pred) 
        
        if debug_level >= 7000:
            debug_file.write("init_frames_probs: self.fast_frames raw probs:\n")
            np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.9f')
            debug_file.write("init_frames_probs: self.fast_frames log probs:\n")
            np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.9f')
            debug_file.write("For each pred, fast frames log probs are:\n")
            for pred_index in range(len(self.predicates)):
                debug_file.write("Predicate {}'s fast frames log probs\n".format(self.predicates[pred_index].pred_name))
                np.savetxt(debug_file, self.predicates[pred_index].fast_frames_probs, fmt='%1.4f')

    def calc_frames_probs_cat(self, debug_level, debug_file, cat_index=None, pred=None, \
                              hyper=False):
        if not hyper:
            fast_frames_count = self.alphas['alpha_frames'] # always have this as base
        else:
            fast_frames_count = self.alphas['alpha_frames_hyper'] # use proposed value during hyperparameter sampling
        if not pred: # doing for category
            if not hyper:
                self.fast_frames_raw_probs[cat_index] = np.zeros((len(self.all_frames))) \
                                                        + fast_frames_count #self.alphas['alpha_frames']
                # check if empty category stats - if so, self.fast_frames_raw_probs already initialized to appropriate alpha value
                if not (cat_index == len(self.categories)):
                    # need to add in category's raw frame prob
                    self.fast_frames_raw_probs[cat_index] += self.categories[cat_index].fast_frames
                    fast_frames_count += self.categories[cat_index].fast_frames   # get count for each frame in 1D array
            else:
                self.hyper_cat_fast_frames_raw_probs[cat_index] = np.zeros((len(self.all_frames))) \
                                                                  + fast_frames_count #self.alphas['alpha_frames']
                if not (cat_index == len(self.categories)):
                    # need to add in category's raw frame prob
                    self.hyper_cat_fast_frames_raw_probs[cat_index] += self.categories[cat_index].fast_frames
                    fast_frames_count += self.categories[cat_index].fast_frames   # get count for each frame in 1D array
  
            if debug_level >= 9000:
                debug_file.write("calc_frames_probs_cat: fast_frames_count for category {} is {}\n".format(cat_index, fast_frames_count))
            if not hyper:
                # divide each element in a row by the row's total   
                self.fast_frames_raw_probs[cat_index] /= self.fast_frames_raw_probs[cat_index].sum(axis=0) 
                # then use this in fast_frames_probs computation (log transform)
                self.fast_frames_probs[cat_index] = (fast_frames_count) * np.log(self.fast_frames_raw_probs[cat_index])

                if debug_level >= 9000:
                    debug_file.write("calc_frames_probs_cat: self.fast_frames raw probs:\n")
                    np.savetxt(debug_file, self.fast_frames_raw_probs, fmt='%1.9f')
                    debug_file.write("self.fast_frames log probs:\n")
                    np.savetxt(debug_file, self.fast_frames_probs, fmt='%1.9f')
            else: # doing for hyperparameter sampling
                 # divide each element in a row by the row's total
                self.hyper_cat_fast_frames_raw_probs[cat_index] /= self.hyper_cat_fast_frames_raw_probs[cat_index].sum(axis=0)
                # then use this in fast_frames_probs computation (log transform)
                self.hyper_cat_fast_frames_probs[cat_index] = (fast_frames_count) * \
                                                               np.log(self.hyper_cat_fast_frames_raw_probs[cat_index])
                if debug_level >= 9000:
                    debug_file.write("calc_frames_probs_cat: self.hyper_cat_fast_frames raw probs:\n")
                    np.savetxt(debug_file, self.hyper_cat_fast_frames_raw_probs, fmt='%1.9f')
                    debug_file.write("self.hyper_cat_fast_frames log probs:\n")
                    np.savetxt(debug_file, self.hyper_cat_fast_frames_probs, fmt='%1.9f')
                
                
            
        else: # doing for pred
            pred.fast_frames_raw_probs = np.zeros((len(self.all_frames))) + self.alphas['alpha_frames'] + pred.fast_frames
            fast_frames_count += pred.fast_frames
            pred.fast_frames_raw_probs /= pred.fast_frames_raw_probs.sum(axis=0) # divide element by total
            pred.fast_frames_probs = (fast_frames_count) * np.log(pred.fast_frames_raw_probs)
            if debug_level >= 9000:
                debug_file.write("calc_frames_probs_cat: Predicate {}'s fast frames raw probs\n".format(pred.pred_name))
                np.savetxt(debug_file, pred.fast_frames_raw_probs, fmt='%1.4f')
            
                        
    def init_sem_prop_probs(self, debug_level, debug_file):
        # initialize one 2D array for each sem_XXX option (sem_subj, sem_obj, sem_iobj)
        # initial size of array = (# categories + new cat poss) x 3 [_gram_subj, _gram_obj, _gram_iobj]
        # sem_subj, sem_obj, sem_iobj
        self.sem_subj_raw_probs = np.zeros((len(self.sampling_p_cat), 3)) 
        self.sem_obj_raw_probs = np.zeros((len(self.sampling_p_cat), 3))
        self.sem_iobj_raw_probs = np.zeros((len(self.sampling_p_cat), 3)) 
        self.sem_subj_probs = np.zeros((len(self.sampling_p_cat), 3)) 
        self.sem_obj_probs = np.zeros((len(self.sampling_p_cat), 3))
        self.sem_iobj_probs = np.zeros((len(self.sampling_p_cat), 3)) 
        for cat_index in range(len(self.categories)+1): # populates empty category too
            if debug_level >= 9000:
                debug_file.write("init_sem_prop_probs: Doing for category {}\n".format(cat_index))
            self.calc_sem_probs_cat(debug_level, debug_file, cat_index) # populates raw_probs and probs                        
        if debug_level >= 7000:
            debug_file.write("self.sem_subj raw totals after normalization (p_multi_prop):\n")
            np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.9f')
            debug_file.write("self.sem_obj raw totals after normalization (p_multi_prop):\n")
            np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.9f')
            debug_file.write("self.sem_iobj raw totals after normalization (p_multi_prop):\n")
            np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.9f')

        # now initialize for all predicates
        for pred_index in range(len(self.predicates)):
            self.predicates[pred_index].sem_subj_raw_probs = np.zeros((3))
            self.predicates[pred_index].sem_obj_raw_probs = np.zeros((3))
            self.predicates[pred_index].sem_iobj_raw_probs = np.zeros((3))
            self.predicates[pred_index].sem_subj_probs = np.zeros((3))
            self.predicates[pred_index].sem_obj_probs = np.zeros((3))
            self.predicates[pred_index].sem_iobj_probs = np.zeros((3))
            self.calc_sem_probs_cat(debug_level, debug_file, pred=self.predicates[pred_index])
            
        if debug_level >= 7000:
            debug_file.write("self.sem_subj log probs:\n")
            np.savetxt(debug_file, self.sem_subj_probs, fmt='%1.9f')
            debug_file.write("self.sem_obj log probs:\n")
            np.savetxt(debug_file, self.sem_obj_probs, fmt='%1.9f')    
            debug_file.write("self.sem_iobj log probs:\n")
            np.savetxt(debug_file, self.sem_iobj_probs, fmt='%1.9f')
            debug_file.write("predicate sem log probs are:\n")
            for pred_index in range(len(self.predicates)):
                debug_file.write("predicate {}:\n".format(self.predicates[pred_index].pred_name))
                debug_file.write("pred.sem_subj log probs:\n")
                np.savetxt(debug_file, self.predicates[pred_index].sem_subj_probs, fmt='%1.4f')
                debug_file.write("\npred.sem_obj log probs:\n")
                np.savetxt(debug_file, self.predicates[pred_index].sem_obj_probs, fmt='%1.4f')
                debug_file.write("\npred.sem_iobj log probs:\n")
                np.savetxt(debug_file, self.predicates[pred_index].sem_iobj_probs, fmt='%1.4f')
                
    def calc_sem_probs_cat(self, debug_level, debug_file, cat_index=None, pred=None, \
                           hyper=False):
        # initialize count variables with pseudocounts
        if hyper:
            sem_subj_count = self.alphas['alpha_sem_subj_hyper']
            sem_obj_count = self.alphas['alpha_sem_obj_hyper']
            sem_iobj_count = self.alphas['alpha_sem_iobj_hyper']
        else:    
            sem_subj_count = self.alphas['alpha_sem_subj']
            sem_obj_count = self.alphas['alpha_sem_obj']
            sem_iobj_count = self.alphas['alpha_sem_iobj']
        
        if not pred: # doing for existing category or empty category
            if debug_level >= 9000:
                debug_file.write("Calculating sem_XXX probs for category with index cat_index {}\n".format(cat_index))

            # add in hyperparameter (including for empty category) (should already be initialized with zeros)
            if hyper:
                self.hyper_cat_sem_subj_raw_probs[cat_index] = np.zeros((3)) + sem_subj_count
                self.hyper_cat_sem_obj_raw_probs[cat_index] = np.zeros((3)) + sem_obj_count 
                self.hyper_cat_sem_iobj_raw_probs[cat_index] = np.zeros((3)) + sem_iobj_count
            else: 
                self.sem_subj_raw_probs[cat_index] = np.zeros((3)) + sem_subj_count
                self.sem_obj_raw_probs[cat_index] = np.zeros((3)) + sem_obj_count 
                self.sem_iobj_raw_probs[cat_index] = np.zeros((3)) + sem_iobj_count
                
            if not (cat_index == len(self.categories)): # not for the empty category
                # initialize with current sem_XXX counts
                if hyper:
                    self.hyper_cat_sem_subj_raw_probs[cat_index] += self.categories[cat_index].sem_subj 
                    self.hyper_cat_sem_obj_raw_probs[cat_index] += self.categories[cat_index].sem_obj
                    self.hyper_cat_sem_iobj_raw_probs[cat_index] += self.categories[cat_index].sem_iobj
                    if debug_level >= 9000:
                        debug_file.write("Before normalizing hyper_cat_sem_XXX_raw probs, they look like this:\n")
                        debug_file.write("self.hyper_cat_sem_subj_raw_probs:\n");
                        np.savetxt(debug_file, self.hyper_cat_sem_subj_raw_probs, fmt='%1.4f')
                        debug_file.write("\nself.hyper_cat_sem_obj_raw_probs:\n");
                        np.savetxt(debug_file, self.hyper_cat_sem_obj_raw_probs, fmt='%1.4f')
                        debug_file.write("\nself.hyper_cat_sem_iobj_raw_probs:\n");
                        np.savetxt(debug_file, self.hyper_cat_sem_iobj_raw_probs, fmt='%1.4f')  
                                      
                else:
                    self.sem_subj_raw_probs[cat_index] += self.categories[cat_index].sem_subj 
                    self.sem_obj_raw_probs[cat_index] += self.categories[cat_index].sem_obj
                    self.sem_iobj_raw_probs[cat_index] += self.categories[cat_index].sem_iobj

                # update count variables
                sem_subj_count += self.categories[cat_index].sem_subj
                sem_obj_count += self.categories[cat_index].sem_obj
                sem_iobj_count += self.categories[cat_index].sem_iobj
                                                        
            # get p_sem_XXX prob: # divide each element in a row by the row's total
            if hyper:
                self.hyper_cat_sem_subj_raw_probs[cat_index] /= self.hyper_cat_sem_subj_raw_probs[cat_index].sum(axis=0) 
                self.hyper_cat_sem_obj_raw_probs[cat_index] /= self.hyper_cat_sem_obj_raw_probs[cat_index].sum(axis=0) 
                self.hyper_cat_sem_iobj_raw_probs[cat_index] /= self.hyper_cat_sem_iobj_raw_probs[cat_index].sum(axis=0)

                if debug_level >= 9000:
                    debug_file.write("After updating hyper_cat_sem_XXX_raw probs, they look like this:\n")
                    debug_file.write("self.hyper_cat_sem_subj_raw_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_subj_raw_probs, fmt='%1.4f')
                    debug_file.write("\nself.hyper_cat_sem_obj_raw_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_obj_raw_probs, fmt='%1.4f')
                    debug_file.write("\nself.hyper_cat_sem_iobj_raw_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_iobj_raw_probs, fmt='%1.4f')            
            else:
                self.sem_subj_raw_probs[cat_index] /= self.sem_subj_raw_probs[cat_index].sum(axis=0) 
                self.sem_obj_raw_probs[cat_index] /= self.sem_obj_raw_probs[cat_index].sum(axis=0) 
                self.sem_iobj_raw_probs[cat_index] /= self.sem_iobj_raw_probs[cat_index].sum(axis=0) 

                if debug_level >= 9000:
                    debug_file.write("After updating sem_XXX_raw probs, they look like this:\n")
                    debug_file.write("self.sem_subj_raw_probs:\n");
                    np.savetxt(debug_file, self.sem_subj_raw_probs, fmt='%1.4f')
                    debug_file.write("\nself.sem_obj_raw_probs:\n");
                    np.savetxt(debug_file, self.sem_obj_raw_probs, fmt='%1.4f')
                    debug_file.write("\nself.sem_iobj_raw_probs:\n");
                    np.savetxt(debug_file, self.sem_iobj_raw_probs, fmt='%1.4f')

            # Note: Include calculation of p_multi_prop ^ # multi_prop here,
            #       which means want log --> # multi_prop * log(p_multi_prop) in each entry
            # So,  each entry currently in sem_xxx_probs[cat_index] =   p_multi_prop
            # sem_xxx_probs[cat_index] = sem_xxx_probs ^ (self.categories[cat_index].sem_xxx + alpha) (element-wise raise to power)
            # --> sem_xxx_probs[cat_index] = np.power(sem_xxx_probs[cat_index], (self.categories[cat_index].sem_xxx + alpha))
            # ---> log: sem_xxx_probs[cat_index] = (self.categories[cat_index].sem_xxx + alpha) * np.log(sem_xxx_probs[cat_index])
            # This will allow them to easily be summed later
            
            # get log tranform - should be for legit category
            if hyper:
                self.hyper_cat_sem_subj_probs[cat_index] = (sem_subj_count) * \
                                                               np.log(self.hyper_cat_sem_subj_raw_probs[cat_index])
                self.hyper_cat_sem_obj_probs[cat_index] = (sem_obj_count) * \
                                                               np.log(self.hyper_cat_sem_obj_raw_probs[cat_index])
                self.hyper_cat_sem_iobj_probs[cat_index] = (sem_iobj_count) * \
                                                               np.log(self.hyper_cat_sem_iobj_raw_probs[cat_index])
                if debug_level >= 9000:
                    debug_file.write("After updating hyper_cat sem_XXX probs, they look like this:\n")
                    debug_file.write("self.hyper_cat_sem_subj_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_subj_probs, fmt='%1.4f')
                    debug_file.write("\nself.hyper_cat_sem_obj_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_obj_probs, fmt='%1.4f')
                    debug_file.write("\nself.hyper_cat_sem_iobj_probs:\n");
                    np.savetxt(debug_file, self.hyper_cat_sem_iobj_probs, fmt='%1.4f')
            else:
                self.sem_subj_probs[cat_index] = (sem_subj_count) * np.log(self.sem_subj_raw_probs[cat_index])
                self.sem_obj_probs[cat_index] = (sem_obj_count) * np.log(self.sem_obj_raw_probs[cat_index])
                self.sem_iobj_probs[cat_index] = (sem_iobj_count) * np.log(self.sem_iobj_raw_probs[cat_index])
                if debug_level >= 9000:
                    debug_file.write("After updating sem_XXX probs, they look like this:\n")
                    debug_file.write("self.sem_subj_probs:\n");
                    np.savetxt(debug_file, self.sem_subj_probs, fmt='%1.4f')
                    debug_file.write("\nself.sem_obj_probs:\n");
                    np.savetxt(debug_file, self.sem_obj_probs, fmt='%1.4f')
                    debug_file.write("\nself.sem_iobj_probs:\n");
                    np.savetxt(debug_file, self.sem_iobj_probs, fmt='%1.4f')
          
        else:
            if debug_level >= 9000:
                debug_file.write("Calculating sem_XXX_probs for predicate {}\n.".format(pred.pred_name))
            pred.sem_subj_raw_probs = np.zeros((3)) + self.alphas['alpha_sem_subj'] # add appropriate hyperparameter to each element
            pred.sem_obj_raw_probs = np.zeros((3)) + self.alphas['alpha_sem_obj'] # add appropriate hyperparameter to each element
            pred.sem_iobj_raw_probs = np.zeros((3)) + self.alphas['alpha_sem_iobj'] # add appropriate hyperparameter to each element
            # initialize with current sem_XXX counts
            pred.sem_subj_raw_probs += pred.sem_subj # initialize with raw count for that category
            pred.sem_obj_raw_probs += pred.sem_obj # initialize with raw count for that category                
            pred.sem_iobj_raw_probs += pred.sem_iobj # initialize with raw count for that category
            # update count variables
            sem_subj_count += pred.sem_subj
            sem_obj_count += pred.sem_obj
            sem_iobj_count += pred.sem_iobj
            # get p_sem_XXX prob
            pred.sem_subj_raw_probs /= pred.sem_subj_raw_probs.sum(axis=0) # divide each element in a row by the row's total
            pred.sem_obj_raw_probs /= pred.sem_obj_raw_probs.sum(axis=0) # divide each element in a row by the row's total
            pred.sem_iobj_raw_probs /= pred.sem_iobj_raw_probs.sum(axis=0) # divide each element in a row by the row's total
            # get log tranform
            pred.sem_subj_probs = (sem_subj_count) * np.log(pred.sem_subj_raw_probs)
            pred.sem_obj_probs = (sem_obj_count) * np.log(pred.sem_obj_raw_probs)
            pred.sem_iobj_probs = (sem_iobj_count) * np.log(pred.sem_iobj_raw_probs)
            
                                          
    def init_bin_prop_probs(self, debug_level, debug_file):
        #   - 2D array: category x +binary property (note: p(-bin_prop) = 1 - p(+bin_prop))

        # initial size of array = (# categories + new cat poss) x (# of binary properties = 3 or 4 if UTAH)
        #  --> ,0 = +anim_gram_subj; ,1 = +anim_gram_obj; ,2 = +anim_gram_iobj ; ,3 = +mvmt
        self.sampling_bin_raw_prop = np.zeros((len(self.sampling_p_cat), 3+self.with_UTAH)) 
        self.sampling_bin_prop = np.zeros((len(self.sampling_p_cat), 3+self.with_UTAH))               
        for cat_index in range(len(self.categories)):
            # calculate bin properties for this cat_index
            self.calc_all_sampling_bin_prop(debug_level, debug_file, cat_index, self.categories[cat_index]) 
        # now initialize new cat
        self.calc_all_sampling_bin_prop(debug_level, debug_file, len(self.sampling_p_cat)-1)

        # now initialize bin prop calculations for all preds
        for pred_index in range(len(self.predicates)):
            # initialize predicate with empty sampling_bin_prop
            self.predicates[pred_index].sampling_bin_raw_prop = np.zeros((3+self.with_UTAH))
            self.predicates[pred_index].sampling_bin_prop = np.zeros((3+self.with_UTAH))
            # then update entries
            self.calc_all_sampling_bin_prop(debug_level, debug_file, pred=self.predicates[pred_index]) 
                                              
        if debug_level >= 7000:
            debug_file.write("init_bin_prop_probs: self.sampling_bin_raw_prop is now \n")
            np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.9f')
            debug_file.write("init_bin_prop_probs: and self.sampling_bin_prop is now \n")
            np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.9f')            
            debug_file.write("and predicate sampling_bin_prop values are:\n")
            for pred_index in range(len(self.predicates)):
                debug_file.write("predicate {}: \n".format(self.predicates[pred_index].pred_name))
                np.savetxt(debug_file, self.predicates[pred_index].sampling_bin_prop, fmt='%1.9f')
                debug_file.write("\n")

    def calc_all_sampling_bin_prop(self, debug_level, debug_file, cat_index=None, category=None, pred=None):
        anim_gram_subj_raw_prob, anim_gram_subj_prob = \
          self.calc_sample_bin_prop(debug_level, debug_file, 0, category, pred)
        anim_gram_obj_raw_prob, anim_gram_obj_prob = \
          self.calc_sample_bin_prop(debug_level, debug_file, 1, category, pred)
        anim_gram_iobj_raw_prob, anim_gram_iobj_prob = \
          self.calc_sample_bin_prop(debug_level, debug_file, 2, category, pred)
        if not pred: # this is the category-level
            self.sampling_bin_raw_prop[cat_index, 0] = anim_gram_subj_raw_prob 
            self.sampling_bin_raw_prop[cat_index, 1] = anim_gram_obj_raw_prob
            self.sampling_bin_raw_prop[cat_index, 2] = anim_gram_iobj_raw_prob
            self.sampling_bin_prop[cat_index, 0] = anim_gram_subj_prob 
            self.sampling_bin_prop[cat_index, 1] = anim_gram_obj_prob
            self.sampling_bin_prop[cat_index, 2] = anim_gram_iobj_prob
        else: # this is for the predicate
            pred.sampling_bin_prop[0] = anim_gram_subj_raw_prob
            pred.sampling_bin_prop[1] = anim_gram_obj_raw_prob
            pred.sampling_bin_prop[2] = anim_gram_iobj_raw_prob  
            pred.sampling_bin_prop[0] = anim_gram_subj_prob
            pred.sampling_bin_prop[1] = anim_gram_obj_prob
            pred.sampling_bin_prop[2] = anim_gram_iobj_prob
        if self.with_UTAH:
            mvmt_raw_prob, mvmt_prob = self.calc_sample_bin_prop(debug_level, debug_file, 3, category, pred)
            if not pred: #category-level
                self.sampling_bin_raw_prop[cat_index, 3] = mvmt_raw_prob 
                self.sampling_bin_prop[cat_index, 3] = mvmt_prob 
            else:
                pred.sampling_bin_raw_prop[3] = mvmt_raw_prob
                pred.sampling_bin_prop[3] = mvmt_prob 
                
        if debug_level >= 9000:
            if not pred:
                debug_file.write("calc_all_sampling_bin_prop, category: self.sampling_bin_raw_prop is now \n")
                np.savetxt(debug_file, self.sampling_bin_raw_prop, fmt='%1.9f')
                debug_file.write("and self.sampling_bin_prop is now \n")
                np.savetxt(debug_file, self.sampling_bin_prop, fmt='%1.9f')
            else:
                debug_file.write("calc_all_sampling_bin_prop, category, predicate {}: \n".format(pred.pred_name))
                np.savetxt(debug_file, pred.sampling_bin_prop, fmt='%1.9f')
                debug_file.write("\n")         

            
    def calc_sample_bin_prop(self, debug_level, debug_file, which_bin, cat=None, pred=None, \
                             hyper=False, hyper_beta0=None, hyper_beta1=None):
        #   - p(+bin_prop in cat_j) = (num_+bin_prop in cat_j + beta_+bin_prop) /
        #                             (num_+/-bin_prop in cat_j + beta_+bin_prop + beta_-bin_prop)
        #                           = (cat.anim_gram_XXX[1] + self.betas['beta1_anim_gram_XXX']) /
        #                             (cat.anim_gram_XXX.sum(axis=0) + self.betas[beta1] + self.betas[beta0]
        #   --> since denominator depends only on instances in that category, only the category
        #       the predicate being sampled comes from will have this number changed; the rest
        #       can stay the same

        # Note: Include calculation of p_bin_prop ^ # bin_prop here,
        #       which means want log --> # bin_prop * log(p_bin_prop) in each entry

        beta0, beta1 = 0.0, 0.0
        # initialize to alpha_frames since this is the basic pseudocount for empty categories (replaced if have actual count)
        # initial values -- update if have category or if sampling and using pred counts
        n_yes_bin, n_all_bin = self.alphas['alpha_frames'], 2*self.alphas['alpha_frames'] 
        
        if which_bin == 0: # anim_gram_subj
            if not hyper:
                beta0, beta1 = self.betas['beta0_anim_gram_subj'], self.betas['beta1_anim_gram_subj']
            else: # grab from passed values
                if debug_level >= 9000:
                    debug_file.write("calc_sample_bin_prop: Using passed proposed hyperparameter values {} and {}\n".\
                                     format(hyper_beta0, hyper_beta1))
                beta0, beta1 = hyper_beta0, hyper_beta1
                   
            if cat:
                n_yes_bin += cat.anim_gram_subj[1]
                n_all_bin += cat.anim_gram_subj.sum(axis=0)
            elif pred: # empty category, but using pred stats
                n_yes_bin += pred.anim_gram_subj[1]
                n_all_bin += pred.anim_gram_subj.sum(axis=0)        
        elif which_bin == 1: # anim_gram_obj
            beta0, beta1 = self.betas['beta0_anim_gram_obj'], self.betas['beta1_anim_gram_obj']
            if cat:
                n_yes_bin += cat.anim_gram_obj[1]
                n_all_bin += cat.anim_gram_obj.sum(axis=0)
            elif pred: # empty category, but using pred stats
                n_yes_bin += pred.anim_gram_obj[1]
                n_all_bin += pred.anim_gram_obj.sum(axis=0)  
        elif which_bin == 2: # anim_gram_iobj
            beta0, beta1 = self.betas['beta0_anim_gram_iobj'], self.betas['beta1_anim_gram_iobj']
            if cat:
                n_yes_bin += cat.anim_gram_iobj[1]
                n_all_bin += cat.anim_gram_iobj.sum(axis=0)
            elif pred: # empty category, but using pred stats
                n_yes_bin += pred.anim_gram_iobj[1]
                n_all_bin += pred.anim_gram_iobj.sum(axis=0)  
        else: # mvmt
            if self.with_UTAH:
                beta0, beta1 = self.betas['beta0_mvmt'], self.betas['beta1_mvmt']
                if cat:
                    n_yes_bin += cat.mvmt[1]
                    n_all_bin += cat.mvmt.sum(axis=0)
                elif pred: # empty category, but using pred stats
                    n_yes_bin += pred.mvmt[1]
                    n_all_bin += pred.mvmt.sum(axis=0)  

        p_bin_prop = (n_yes_bin + beta1)/(n_all_bin + beta0 + beta1)
        # p_yes ^ # yes + (1-p_yes) ^ # no =
        # (p_bin_prop ^ n_yes_bin) * ((1-p_bin_prop)^(n_all_bin - n_yes_bin))
        # --> log = (n_yes_bin * log(p_bin_prop)) + ((n_all_bin - n_yes_bin)* log(1-p_bin_prop))
        if debug_level >= 9000:
            debug_file.write("calc_sample_bin_prop: Cat {}, Calculating (n_yes_bin * log(p_bin_prop)) + ((n_all_bin - n_yes_bin)*log(1-p_bin_prop)\n".\
                             format(cat))
            debug_file.write("({} * log({})) + (({} - {})*log(1-{})\n".format(n_yes_bin, p_bin_prop, n_all_bin, n_yes_bin, p_bin_prop))
        return (p_bin_prop, ((n_yes_bin * log(p_bin_prop)) + ((n_all_bin - n_yes_bin)*log(1-p_bin_prop))))
            
    def set_annealing_schedule(self, debug_level, debug_file, output_file):
        # from Stella Frank's code, which comes from Sharon Goldwater's code:
        # lower over 20 increments
        if self.iterations < 20:
            output_file.write("\nNot enough iterations to anneal: not annealing, temp always = 1\n")
            self.with_annealing = False
        else:
            self.annealing_schedule = [self.annealing['stop_temp'] + x* \
                                       ((self.annealing['start_temp']-self.annealing['stop_temp'])/19) \
                                       for x in range(20)] 
            self.annealing_schedule.reverse()
            output_file.write("\nAnnealing schedule set from {} down to {} over 20 steps\n".format(\
                              self.annealing['start_temp'], self.annealing['stop_temp']))                                                                        
            if debug_level >= 7000:
                debug_file.write("set_annealing_schedule: annealing schedule is\n")
                print_array(self.annealing_schedule, debug_file)
    
    def initialize_categories(self, debug_level, debug_file):

        self.cat_count = 0 # number of categories currently, initialized to 0
        # go through each predicate and assign a category - use numpy array for faster computation
        self.cat_freq = array([self.gamma_c]) # first entry initialized to gamma_c value
        # sort self.all_frames so can use to construct fast_frames entries
        self.all_frames.sort()
        if debug_level >= 9000:
            debug_file.write("initialize_categories: self.all_frames now looks like this\n")
            print_array(self.all_frames, debug_file)
        
        for pred_index in range(len(self.predicates)):
            # Prob(class c_j) = (# in c_j already + gamma_c)/(# of predicates total + C*gamma_c)
            #      where c_j is the category assignment and
            #      C is the number of categories that currently exist.
            # First predicate is assigned to category 0 by default.
            if debug_level >= 7000:
                debug_file.write("Now on pred_index {} of {}\n".format(pred_index, len(self.predicates)-1))
            
            if pred_index == 0:
                self.predicates[pred_index].set_category(0)
                if debug_level >= 5000:
                    debug_file.write("initialize_categories: set first pred category {} to {}\n".\
                                     format(self.predicates[pred_index].pred_name, \
                                            self.predicates[pred_index].category))
                self.cat_freq[0] += 1 # update cat 0 entry by 1
                self.cat_count += 1 # now have one category
                # add new gamma_c value for possible new category
                self.cat_freq = np.append(self.cat_freq, self.gamma_c) 
                
                if debug_level >= 5000:
                    debug_file.write("initialize_categories: cat 0 frequency is now {}\n".\
                                     format(self.cat_freq[0]))
                    debug_file.write("cat_freq array now \n")
                    print_array(self.cat_freq, debug_file)

                # now create Category 0 and fill its counts with this predicate's values
                # Binary properties:
                #   anim_gram_subj, inanim_gram_subj
                #   anim_gram_obj, inanim_gram_obj
                #   anim_gram_iobj, inanim_gram_iobj
                # if UTAH:
                #   mvmt
                # Multinomial properties:
                # if no UTAH:
                #      sem_subj (as gram_subj, gram_obj, or gram_ind_obj)
                #      sem_obj (as gram_subj, gram_obj, gram_ind_obj)
                #      sem_iobj (as gram_subj, gram_obj, gram_ind_obj)
                # frames (as TO V PP, NP V NP, NP Aux V-en, ...)
                # also create fast_frames and mvmt entries inside predicate, using self.all_frames and sem_xxx arrays
                new_cat = Category(debug_level, debug_file, \
                                   self.all_frames, self.predicates[pred_index],\
                                   self.with_UTAH, 0, True)
                # now add new_cat to list of categories -- it's the first one                 
                self.categories = [new_cat]
                if debug_level >= 9000:
                    debug_file.write("initialize_categories: after first category, self.categories is\n")
                    for cat in self.categories:
                        cat.print_cat_info(debug_file, self.with_UTAH)

            else: # use probabilistic assignment
                # roll weighted len(self.cat_freq) sided die to determine category assignment
                pred_cat = get_rand_weighted(self.cat_freq, debug_level, debug_file)
                # assign this category to this predicate
                self.predicates[pred_index].set_category(pred_cat)
                if debug_level >= 5000:
                    debug_file.write("initialize_categories: set pred {} to category {}\n".\
                                     format(self.predicates[pred_index].pred_name, \
                                            self.predicates[pred_index].category))
                # if new category, update cat_count & cat_freq with new entry for future new category
                if pred_cat == (len(self.cat_freq)-1):
                    if debug_level >= 7000:
                        debug_file.write("New category - updating cat_count and cat_freq\n")
                    self.cat_count += 1
                    self.cat_freq = np.append(self.cat_freq, self.gamma_c)
                     # now create new Category and fill its counts with this predicate's values
                    new_cat = Category(debug_level, debug_file,\
                                       self.all_frames, self.predicates[pred_index],\
                                       self.with_UTAH, pred_cat, True)
                    self.categories.append(new_cat) # add new category to self.categories
                    if debug_level >= 9000:
                        debug_file.write("initialize_categories: after new category, self.categories is\n")
                        for cat in self.categories:
                            cat.print_cat_info(debug_file, self.with_UTAH)
                                       
                else: # update existing Category pred_cat with this pred's info
                    self.categories[pred_cat].add_pred_info(debug_level, debug_file, \
                                                       self.all_frames, self.predicates[pred_index],\
                                                       self.with_UTAH, True)

                         
                # in general, update cat_freq with this pred (+1)
                self.cat_freq[pred_cat] += 1
                
                    
                if debug_level >= 5000:
                    debug_file.write("initialize_categories: cat {} frequency is now {}\n".\
                                     format(pred_cat, self.cat_freq[pred_cat]))
                    debug_file.write("cat_freq array now \n")
                    print_array(self.cat_freq, debug_file) 

        if debug_level >= 5000: # check to see how predicates were initialized
            debug_file.write("initialize_categories: after initialization, category membership is\n")
            for pred in sorted(self.predicates, key = lambda pred: pred.category): # by category
                debug_file.write("Category {}: pred {}\n".format(pred.category, pred.pred_name))
            for cat in self.categories: # check to see what categories look like
                debug_file.write("Category {} info:\n".format(cat.cat_num))
                cat.print_cat_info(debug_file, self.with_UTAH)    
                                
        if debug_level >= 10000: # sanity check that we're rolling weighted die appropriately
            sanity_list = array([0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2])
            check = stats.rv_discrete(values=(np.arange(len(sanity_list)), sanity_list))
            check_list = check.rvs(size=50)
            print_array(check_list, debug_file)                
        
    def read_pred_entries(self, input_file, debug_level, debug_file):
        if debug_level >= 9000:
            debug_file.write("Opening input_file {} for reading\n".format(input_file))

        self.predicates = [] # to hold collection of predicates read in from input file, currently empty
        self.all_frames = [] # to hold collection of frames from all predicates, used to construct fast_frames numpy array
                             # for each predicate and each category
        
        with open(input_file, 'r') as infile:
            if debug_level >= 5000:
                debug_file.write("read_pred_entries: reading lines in {}\n".format(input_file))

            inline = infile.readline()
            while inline:
                if debug_level >= 9000:
                    debug_file.write("\ninline is {}".format(inline))
                # is it the beginning of an entry?
                #HEAR                
                pattern = re.compile('^[A-Z\-]+$') # all caps and maybe a hyphen
                match = pattern.search(inline) # is that what inline has?
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched entry beginning {}\n".format(match.group()))
                    # this is the predicate name
                    pred_name = match.group()

                inline = infile.readline() # get next line
                # anim-gram-subj vs. inanim: 101 1
                pattern = re.compile('^anim-gram-subj vs. inanim: (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched anim-gram-subj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: anim-gram-subj = {}, inanim-gram-subj = {}\n".format(match.group(1), match.group(2)))
                    # these are the anim-gram-subj and inanim-gram-subj
                    anim_gram_subj, inanim_gram_subj = match.group(1), match.group(2)

                inline = infile.readline()
                # anim-gram-obj vs. inanim: 50 50
                pattern = re.compile('^anim-gram-obj vs. inanim: (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched anim-gram-obj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: anim-gram-obj = {}, inanim-gram-obj = {}\n".format(match.group(1), match.group(2)))
                    # these are the anim-gram-obj and inanim-gram-obj
                    anim_gram_obj, inanim_gram_obj = match.group(1), match.group(2)

                inline = infile.readline()
                # anim-gram-iobj vs. inanim: 5 2
                pattern = re.compile('^anim-gram-iobj vs. inanim: (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched anim-gram-iobj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: anim-gram-iobj = {}, inanim-gram-iobj = {}\n".format(match.group(1), match.group(2)))
                    # these are the anim-gram-iobj and inanim-gram-iobj
                    anim_gram_iobj, inanim_gram_iobj = match.group(1), match.group(2)

                inline = infile.readline()
                # sem-subj as gram-subj vs. gram-obj vs. gram-iobj: 100 2 0 
                pattern = re.compile('^sem-subj as gram-subj vs. gram-obj vs. gram-iobj: (\d+) (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched sem-subj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: gram-subj = {}, gram-obj = {}, gram-iobj = {}\n".format(match.group(1), match.group(2), match.group(3)))
                    # these are the sem-subj gram-subj, gram-obj, and gram-iobj
                    sem_subj_gram_subj, sem_subj_gram_obj, sem_subj_gram_iobj = \
                      match.group(1), match.group(2), match.group(3)

                inline = infile.readline()
                # sem-obj as gram-subj vs. gram-obj vs. gram-iobj: 2 100 0 
                pattern = re.compile('^sem-obj as gram-subj vs. gram-obj vs. gram-iobj: (\d+) (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 5000:
                        debug_file.write("read_pred_entries: matched sem-obj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: gram-subj = {}, gram-obj = {}, gram-iobj = {}\n".format(match.group(1), match.group(2), match.group(3)))
                    # these are the sem-obj gram-subj, gram-obj, and gram-iobj
                    sem_obj_gram_subj, sem_obj_gram_obj, sem_obj_gram_iobj = \
                      match.group(1), match.group(2), match.group(3)

                inline = infile.readline()
                # sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: 0 0 7 
                pattern = re.compile('^sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: (\d+) (\d+) (\d+)') 
                match = pattern.search(inline)
                if match:
                    if debug_level >= 9000:
                        debug_file.write("read_pred_entries: matched sem-iobj line: {}\n".format(match.group()))
                        debug_file.write("read_pred_entries: gram-subj = {}, gram-obj = {}, gram-iobj = {}\n".format(match.group(1), match.group(2), match.group(3)))
                    # these are the sem-iobj gram-subj, gram-obj, and gram-iobj
                    sem_iobj_gram_subj, sem_iobj_gram_obj, sem_iobj_gram_iobj = \
                      match.group(1), match.group(2), match.group(3)

                inline = infile.readline()                      
                #frames:
                pattern = re.compile('frames\:')
                match = pattern.search(inline)
                if match:
                    pred_frames = {} # initialize frames hash
                    inline = infile.readline()
                    end_pattern = re.compile('\*\*\*')
                    while not end_pattern.search(inline):
                        #TO V PP: 50
                        #NP V NP: 40
                        #NP Aux V-en: 2
                        #NP V CP-that: 8
                        #NP V CP-null: 2
                        #***
                        pattern = re.compile('^([\w \-\']+)\: (\d+)')
                        match = pattern.search(inline)
                        if match:
                            if debug_level >= 5000:
                                debug_file.write("read_pred_entries: matched frame line: {}\n".format(match.group()))
                                debug_file.write("read_pred_entries: frame = {}, freq = {}\n".format(match.group(1), match.group(2)))
                            # this is a frame - add to frames list
                            pred_frames[match.group(1)] = int(match.group(2))
                            # add frame to self.all_frames if not there already
                            if match.group(1) not in self.all_frames:
                                self.all_frames.append(match.group(1))
                                if debug_level >= 9000:
                                    debug_file.write("read_pred_entries: Adding {} to self.all_frames\n".format(match.group(1)))
                            if debug_level >= 9000:
                                debug_file.write("read_pred_entries: Now self.all_frames is \n")
                                print_array(self.all_frames, debug_file)
                            
                        inline = infile.readline()  # get next line       
                    if debug_level >= 5000:
                        debug_file.write("read_pred_entries: frames are\n")
                        print_hash(pred_frames, debug_file)
                        
                            
                # then create Predicate instance populated with the gleaned values
                new_pred = Predicate(debug_file, debug_level, pred_name, \
                                     int(anim_gram_subj), int(inanim_gram_subj), \
                                     int(anim_gram_obj), int(inanim_gram_obj), \
                                     int(anim_gram_iobj), int(inanim_gram_iobj), \
                                     int(sem_subj_gram_subj), int(sem_subj_gram_obj), int(sem_subj_gram_iobj), \
                                     int(sem_obj_gram_subj), int(sem_obj_gram_obj), int(sem_obj_gram_iobj), \
                                     int(sem_iobj_gram_subj), int(sem_iobj_gram_obj), int(sem_iobj_gram_iobj), \
                                     pred_frames)


                # add created Predicate to list of Predicates inside PredLearner
                self.predicates.append(new_pred)
                
                # read in next line
                inline = infile.readline()   

        if debug_level >= 3500:
            debug_file.write("read_pred_entries: After reading in all the predicate entries...\n")
            self.print_predicates(debug_file)

    # print all predicates            
    def print_predicates(self, output_file):
        for pred in sorted(self.predicates, key = lambda pred: pred.pred_name): # alphabetical by pred_name
            pred.print_pred_info(output_file)

    # print log posterior & cat information
    def print_learner_stats(self, debug_level, debug_file, output_file):
        # print out basic stats
        # (1) current number of categories - exclude deleted categories and empty category, since just there for sampling
        output_file.write("Current number of categories = {}\n".format(np.count_nonzero(self.cat_freq)-1))
        # (2) how many categories changed their members this iteration (use self.category_changed)
        output_file.write("Categories that changed members = {}\n".format(np.count_nonzero(self.category_changed)))
        # (3) current log posterior
        output_file.write("Current log posterior = {}\n".format(self.log_posterior()))
        # (4) if sampling hyperparameters, current hyperparameters
        if self.sample_hyper:
          output_file.write("Current hyperparameters are\n")
          self.print_hyperparams(output_file)              
            
   # for printing hyperparameters and annealing parameters to output file
    def print_hyperparams_annealing(self, output_file):
        if self.with_UTAH:
            output_file.write("Using UTAH assumption\n\n")
        self.print_hyperparams(output_file)
        if self.with_annealing:
            output_file.write("Using annealing. Annealing values are\n")
            for anneal_k, anneal_v in sorted(self.annealing.items()):
                output_file.write("{}: {}\n".format(anneal_k, anneal_v))
        else:
            output_file.write("No annealing\n")         
        output_file.write("")

    # print hyperparams
    def print_hyperparams(self, output_file):
        output_file.write("Hyperparameter values are:\n")
        output_file.write("gamma_c: {}\n".format(self.gamma_c)) 
        for beta_k, beta_v in sorted(self.betas.items()):
            output_file.write("{}: {}\n".format(beta_k, beta_v))    
        for alpha_k, alpha_v in sorted(self.alphas.items()):
            output_file.write("{}: {}\n".format(alpha_k, alpha_v))
        output_file.write("\n")               
                
    # constructor for PredLearner
    def __init__(self, \
                 debug_file, output_file, \
                 debug_level, with_UTAH, gamma_c, \
                 beta0_anim_gram_subj, beta1_anim_gram_subj,\
                 beta0_anim_gram_obj, beta1_anim_gram_obj, \
                 beta0_anim_gram_iobj, beta1_anim_gram_iobj, \
                 beta0_mvmt, beta1_mvmt, \
                 alpha_frames, \
                 alpha_sem_subj, alpha_sem_obj, alpha_sem_iobj, \
                 with_annealing, start_temp, stop_temp,
                 iterations, sample_hyper, eval_iterations):
                
        # assign hyperparameter values
        # hyperparameters:
        #   categories: gamma_c (default = 0.001)
        self.gamma_c = gamma_c
        
        #   binary: beta1_anim_gram_subj (+), beta0_anim_gram_subj (-)
        #           beta1_anim_gram_obj (+), beta0_anim_gram_obj (-)
        #           beta1_anim_gram_iobj (+), beta0_anim_gram_iobj (-)
        # defaults = 0.001 (from command line args & defaults)
        self.betas = {}
        self.betas['beta1_anim_gram_subj'], self.betas['beta0_anim_gram_subj'] = \
          beta1_anim_gram_subj, beta0_anim_gram_subj
        self.betas['beta1_anim_gram_obj'], self.betas['beta0_anim_gram_obj'] = \
          beta1_anim_gram_obj, beta0_anim_gram_obj
        self.betas['beta1_anim_gram_iobj'], self.betas['beta0_anim_gram_iobj'] = \
          beta1_anim_gram_iobj, beta0_anim_gram_iobj
        
        self.alphas = {}
        # no matter what, initialize frames hyperparameter      
        self.alphas['alpha_frames'] = alpha_frames
        #           if UTAH: beta1_mvmt (+), beta0_mvmt (-)
        self.with_UTAH = with_UTAH
        if self.with_UTAH:
            self.betas['beta1_mvmt'], self.betas['beta0_mvmt'] = beta1_mvmt, beta0_mvmt
        else: # initialize alpha_sem_XXX hyperparameters
            #   multinomial: alpha_sem_subj (if not UTAH)
            #                alpha_sem_obj  (if not UTAH)
            #                alpha_sem_iobj (if not UTAH)
            #                alpha_frames
            # defaults = 0.001 from command line args  
            self.alphas['alpha_sem_subj'], self.alphas['alpha_sem_obj'],self.alphas['alpha_sem_iobj'] = \
              alpha_sem_subj, alpha_sem_obj, alpha_sem_iobj
       

        self.with_annealing = with_annealing
        if self.with_annealing:      
            # assign annealing schedule parameters
            # annealing schedule parameters:
            #   start_temp (2 in GG2007)
            #   stop_temp (0.8 in GG2007)
            self.annealing = {}
            self.annealing['start_temp'], self.annealing['stop_temp'] = start_temp, stop_temp

        self.iterations = iterations
        self.sample_hyper = sample_hyper
        self.eval_iterations = eval_iterations
            
        # sanity-check for initialized values
        if(debug_level >= 9000):
            debug_file.write("DEBUG: After __init__ function\n")
            self.print_hyperparams_annealing(debug_file)
        # general output
        self.print_hyperparams_annealing(output_file)    
        output_file.write("\nSampling for {} iterations\n".format(self.iterations))
        output_file.write("Evaluate every {} iterations\n".format(self.eval_iterations))
        output_file.write("Sampling hyperparameters? {}\n".format(self.sample_hyper))
        

#################################### 
# class that holds single category and associated counts
class Category:
    # adding element info
    # if during initialization, pred info also has to be set up for
    #   frames (always) and mvmt (if UTAH)
    def add_pred_info(self, debug_level, debug_file, \
                      all_frames, pred,\
                      with_UTAH, during_init):
        # one more pred type than before
        self.num_preds += 1
        # add this predicate to list of predicates the category has
        self.predicates.append(pred)
        self.add_binary_properties(pred)
        if during_init: # initialize pred.mvmt (if UTAH) and pred.fast_frames
            self.init_multinomial_properties(debug_level, debug_file, pred, all_frames, with_UTAH)
        self.add_multinomial_properties(pred, all_frames, with_UTAH)
        if debug_level >= 7000:
            debug_file.write("add_pred_info: after adding info in for pred:\n".format(pred.pred_name))
            self.print_cat_info(debug_file, with_UTAH)

            
    def init_multinomial_properties(self, debug_level, debug_file, pred, all_frames, with_UTAH):
        if with_UTAH:
            #   mvmt - construct from pred.sem_subj, pred.sem_obj, pred.sem_iobj
            no_mvmt, yes_mvmt = self.calc_mvmt(debug_level, debug_file, pred)
            # make temp_mvmt which can be added to self.mvmt and initialize pred's fast_frames
            temp_mvmt = np.zeros(2, dtype=np.int)
            temp_mvmt[0] = no_mvmt
            temp_mvmt[1] = yes_mvmt
            #   then create mvmt numpy array in pred as well
            pred.set_mvmt(temp_mvmt)

        # use all_frames to set up temp_fast_frames numpy array with zeros, 1 for each frame in all_frames
        temp_fast_frames = np.zeros(len(all_frames), dtype=np.int)
        # for each frame in pred.pred_frames, initialize appropriate index in temp_fast_frames
        #   using all_frames as a reference of what index corresponds to what frame
        for p_frame, p_frame_freq in pred.pred_frames.items():
            # find index of p_frame = all_frames.index(p_frame)
            temp_fast_frames[all_frames.index(p_frame)] = p_frame_freq
            
        # then create fast frames nump array in pred as well from temp_fast_frames
        pred.set_fast_frames(temp_fast_frames)                     
            
    def add_multinomial_properties(self, pred, all_frames, with_UTAH):
        if with_UTAH:
            self.mvmt += pred.mvmt
            
        else: # no_UTAH
            #      sem_subj (as gram_subj, gram_obj, or gram_ind_obj)
            #      sem_obj (as gram_subj, gram_obj, gram_ind_obj)
            #      sem_iobj (as gram_subj, gram_obj, gram_ind_obj)
            self.sem_subj += pred.sem_subj
            self.sem_obj += pred.sem_obj
            self.sem_iobj += pred.sem_iobj    
            
        # frames (as TO V PP, NP V NP, NP Aux V-en, ...)
        # add pred.fast_frames counts to self.fast_frames
        self.fast_frames += pred.fast_frames
            
                
    def add_binary_properties(self, pred):
        # Binary properties:
        #   anim_gram_subj, inanim_gram_subj
        #   anim_gram_obj, inanim_gram_obj
        #   anim_gram_iobj, inanim_gram_iobj
        # initialize all to pred's counts      
        self.anim_gram_subj += pred.anim_gram_subj
        self.anim_gram_obj += pred.anim_gram_obj
        self.anim_gram_iobj += pred.anim_gram_iobj                   
    
    def calc_mvmt(self, debug_level, debug_file, pred):
        #   -mvmt [0] = sem_subj[0] + sem_obj[1] + sem_iobj[2]
        #   +mvmt [1] = sem_subj[1] and [2], sem_obj[0] and [2], sem_iobj[0] and [1]
        #             = sum of sem_subj + sem_obj + sem_iobj - mvmt[0]
        no_mvmt = pred.sem_subj[0] + pred.sem_obj[1] + pred.sem_iobj[2]
        if debug_level >= 10000:
            debug_file.write("no_mvmt = {} + {} + {} = {}\n".format(pred.sem_subj[0], pred.sem_obj[1], \
                                                                    pred.sem_iobj[2], no_mvmt))
        # yes_mvmt calculation
        yes_mvmt = pred.sem_subj.sum(axis=0) + pred.sem_obj.sum(axis=0) + \
              pred.sem_iobj.sum(axis=0) - no_mvmt
        return no_mvmt, yes_mvmt      
    
    def __init__(self, debug_level, debug_file, \
                 all_frames, pred, \
                 with_UTAH, cat_num, during_init=False):
        # category number
        self.cat_num = cat_num       
        # how many pred types the category has - initialize to 1
        self.num_preds = 1
        # list of predicates the category has - initialize to pred passed
        self.predicates = [pred]
        # Binary properties:
        #   anim_gram_subj, inanim_gram_subj
        #   anim_gram_obj, inanim_gram_obj
        #   anim_gram_iobj, inanim_gram_iobj
        self.anim_gram_subj = np.zeros(len(pred.anim_gram_subj))
        self.anim_gram_obj = np.zeros(len(pred.anim_gram_obj))
        self.anim_gram_iobj = np.zeros(len(pred.anim_gram_iobj))
        
        # initialize all to pred's counts      
        self.anim_gram_subj += pred.anim_gram_subj
        self.anim_gram_obj += pred.anim_gram_obj
        self.anim_gram_iobj += pred.anim_gram_iobj    

        if during_init: # need to calculate mvmt and fast_frames and set pred.mvmt, pred_fast_frames
            self.init_multinomial_properties(debug_level, debug_file, pred, all_frames, with_UTAH)
            
        if with_UTAH:
            self.mvmt = np.zeros(len(pred.mvmt))
            self.mvmt += pred.mvmt
        else: # no_UTAH
            self.sem_subj = np.zeros(len(pred.sem_subj))
            self.sem_obj = np.zeros(len(pred.sem_obj))
            self.sem_iobj = np.zeros(len(pred.sem_iobj))
            #      sem_subj (as gram_subj, gram_obj, or gram_ind_obj)
            #      sem_obj (as gram_subj, gram_obj, gram_ind_obj)
            #      sem_iobj (as gram_subj, gram_obj, gram_ind_obj)
            self.sem_subj += pred.sem_subj
            self.sem_obj += pred.sem_obj
            self.sem_iobj += pred.sem_iobj    
            
        # frames (as TO V PP, NP V NP, NP Aux V-en, ...)
        self.fast_frames = np.zeros(len(pred.fast_frames))
        self.fast_frames += pred.fast_frames
        # use all_frames to set up fast_frames numpy array with zeros, 1 for each frame in all_frames
        #self.fast_frames = np.zeros(len(all_frames), dtype=np.int)
        # for each frame in pred.pred_frames, initialize appropriate index in fast_frames
        #   using all_frames as a reference of what index corresponds to what frame
        #for p_frame, p_frame_freq in pred.pred_frames.items():
            # find index of p_frame = all_frames.index(p_frame)
            #self.fast_frames[all_frames.index(p_frame)] = p_frame_freq
                   
        if debug_level >= 7000:
            debug_file.write("Category info after initialization:\n")
            self.print_cat_info(debug_file, with_UTAH)                 

    def print_cat_info(self, output_file, with_UTAH):
        output_file.write("Category number {} ".format(self.cat_num))
        output_file.write("has {} predicates:\n".format(self.num_preds))
        for pred in sorted(self.predicates, key = lambda pred: pred.pred_name): # by name
                output_file.write("predicate: {}\n".format(pred.pred_name))
        output_file.write("anim_gram_subj vs. inanim: {} {}\n".format(self.anim_gram_subj[1], \
                                                                      self.anim_gram_subj[0]))
        output_file.write("anim_gram_obj vs. inanim: {} {}\n".format(self.anim_gram_obj[1], \
                                                                      self.anim_gram_obj[0]))
        output_file.write("anim_gram_iobj vs. inanim: {} {}\n".format(self.anim_gram_iobj[1], \
                                                                      self.anim_gram_iobj[0]))
        if with_UTAH:
            output_file.write("+mvmt vs. -mvmt: {} {}\n".format(self.mvmt[1], self.mvmt[0]))
        else:
            output_file.write("sem_subj as gram_subj vs. gram_obj vs. gram_iobj: {} {} {}\n".format(\
                              self.sem_subj[0], self.sem_subj[1], self.sem_subj[2]))
            output_file.write("sem_obj as gram_subj vs. gram_obj vs. gram_iobj: {} {} {}\n".format(\
                              self.sem_obj[0], self.sem_obj[1], self.sem_obj[2]))
            output_file.write("sem_iobj as gram_subj vs. gram_obj vs. gram_iobj: {} {} {}\n".format(\
                              self.sem_iobj[0], self.sem_iobj[1], self.sem_iobj[2]))
        output_file.write("frame frequencies:\n")
        #print_array(self.fast_frames, output_file)
        np.savetxt(output_file, self.fast_frames, fmt = '%1.3f')                                                                                         
                                                                                                                                                    
####################################         
# class that holds single predicate and associated statistics
class Predicate:
    def set_category(self, cat_num):
        self.category = cat_num
        
    def set_mvmt(self, mvmt):
        # for UTAH assumption
        self.mvmt = mvmt

    def set_fast_frames(self, fast_frames): # based on all_frames list
        self.fast_frames = fast_frames       
                 
    def __init__(self, debug_file, debug_level, pred_name, \
                 anim_gram_subj, inanim_gram_subj, \
                 anim_gram_obj, inanim_gram_obj, \
                 anim_gram_iobj, inanim_gram_iobj, \
                 sem_subj_gram_subj, sem_subj_gram_obj, sem_subj_gram_iobj, \
                 sem_obj_gram_subj, sem_obj_gram_obj, sem_obj_gram_iobj, \
                 sem_iobj_gram_subj, sem_iobj_gram_obj, sem_iobj_gram_iobj, \
                 pred_frames):
        self.pred_name = pred_name
        # anim_gram_subj[0] = -anim, [1] = +anim
        self.anim_gram_subj = array([inanim_gram_subj, anim_gram_subj])         
        # anim_gram_obj[0] = -anim, [1] = +anim
        self.anim_gram_obj = array([inanim_gram_obj, anim_gram_obj])
        # anim_gram_iobj[0] = -anim, [1] = +anim 
        self.anim_gram_iobj = array([inanim_gram_iobj, anim_gram_iobj])
        # sem_subj[0] = gram_subj, [1] = gram_obj, [2] = gram_iobj
        self.sem_subj = array([sem_subj_gram_subj,sem_subj_gram_obj, sem_subj_gram_iobj]) 
        # sem_obj[0] = gram_subj, [1] = gram_obj, [2] = gram_iobj
        self.sem_obj = array([sem_obj_gram_subj,sem_obj_gram_obj, sem_obj_gram_iobj])
        # sem_iobj[0] = gram_subj, [1] = gram_obj, [2] = gram_iobj
        self.sem_iobj = array([sem_iobj_gram_subj,sem_iobj_gram_obj, sem_iobj_gram_iobj])
        # keep this one as regular dictionary/hash -- will have separate numpy array one for fast computation
        self.pred_frames = pred_frames

        if debug_level >= 4000:
            self.print_pred_info(debug_file)

                  
    def print_pred_info(self, output_file):
       output_file.write("{}\n".format(self.pred_name))
       if hasattr(self, 'category'):
           output_file.write("in category {}\n".format(self.category))
       output_file.write("anim_gram_subj vs. inanim_gram_subj: {} {}\n".format(self.anim_gram_subj[1], \
                                                                               self.anim_gram_subj[0]))
       output_file.write("anim_gram_obj vs. inanim_gram_obj: {} {}\n".format(self.anim_gram_obj[1], \
                                                                               self.anim_gram_obj[0]))            
       output_file.write("anim_gram_iobj vs. inanim_gram_iobj: {} {}\n".format(self.anim_gram_iobj[1], \
                                                                               self.anim_gram_iobj[0])) 
       output_file.write("sem_subj: gram_subj vs.gram_obj vs. gram_iobj: {} {} {}\n".\
                         format(self.sem_subj[0], self.sem_subj[1], self.sem_subj[2]))
       output_file.write("sem_obj: gram_subj vs.gram_obj vs. gram_iobj: {} {} {}\n".\
                         format(self.sem_obj[0], self.sem_obj[1], self.sem_obj[2]))
       output_file.write("sem_iobj: gram_subj vs.gram_obj vs. gram_iobj: {} {} {}\n".\
                         format(self.sem_iobj[0], self.sem_iobj[1], self.sem_iobj[2]))
       output_file.write("frames:\n")
       print_hash(self.pred_frames, output_file)
       if hasattr(self, 'mvmt'): # only print this out if it's been initialized already
           output_file.write("mvmt (-, then +): ")
           np.savetxt(output_file, self.mvmt, fmt = '%1.9f')
       if hasattr(self, 'fast_frames'): # only print this out if it's been initialized already
           output_file.write("fast_frames: ")
           np.savetxt(output_file, self.fast_frames, fmt = '%1.9f')
       if hasattr(self, 'sem_subj_raw_probs'):
           output_file.write("sem_subj_raw_probs: \n")
           np.savetxt(output_file, self.sem_subj_raw_probs, fmt = '%1.9f')
       if hasattr(self, 'sem_obj_raw_probs'):
           output_file.write("sem_obj_raw_probs: \n")
           np.savetxt(output_file, self.sem_obj_raw_probs, fmt = '%1.9f')
       if hasattr(self, 'sem_iobj_raw_probs'):
           output_file.write("sem_iobj_raw_probs: \n")
           np.savetxt(output_file, self.sem_iobj_raw_probs, fmt = '%1.9f')
       if hasattr(self, 'sem_subj_probs'):
           output_file.write("sem_subj_probs: \n")
           np.savetxt(output_file, self.sem_subj_probs, fmt = '%1.9f')
       if hasattr(self, 'sem_obj_probs'):
           output_file.write("sem_obj_probs: \n")
           np.savetxt(output_file, self.sem_obj_probs, fmt = '%1.9f')
       if hasattr(self, 'sem_iobj_probs'):
           output_file.write("sem_iobj_probs: \n")
           np.savetxt(output_file, self.sem_iobj_probs, fmt = '%1.9f')
       if hasattr(self, 'fast_frame_raw_probs'):
           output_file.write("fast_frame_raw_probs: \n")
           np.savetxt(output_file, self.fast_frame_raw_probs, fmt = '%1.9f')
       if hasattr(self, 'fast_frame_probs'):
           output_file.write("fast_frame_probs: \n")
           np.savetxt(output_file, self.fast_frame_probs, fmt = '%1.9f')
       if hasattr(self, 'log_p_binary'):
           output_file.write("log_p_binary: {} \n".format(self.log_p_binary))
       if hasattr(self, 'log_p_multinomial'):
           output_file.write("log_p_multinomial: {} \n".format(self.log_p_multinomial))
       output_file.write("\n")

####################################               
# function for pulling out random index from weighted list
# ex: roll weighted len(self.cat_freq) sided die to determine category assignment
# For log probs: Add in optional variable to un-log after normalizing if sending in log probabilities (which cat probs will be)
def get_rand_weighted(weighted_list, debug_level, debug_file, logged=False): 
    ret_index = -1000 # initialize to bogus index so we can sanity check this worked

    # if in log form, need to apply exp to get back actual values
    # to help avoid underflow, add fixed value = the smallest neg value
    # (this is equivalent to multiplying by exp(that value) in normal space - 
    #  so it washes out once we normalize)
    # this will help move numbers out of underflow territory, but won't completely prevent it
    if logged:
        max_elem = weighted_list[np.argmax(weighted_list)]
        if (debug_level >= 8000):
            debug_file.write("get_rand_weighted: Before adding {} to help prevent underflow, weighted_list is\n".format(\
                             max_elem))
            np.savetxt(debug_file, weighted_list, fmt='%1.4f') 
        weighted_list -= max_elem
        if (debug_level >= 8000):
            debug_file.write("get_rand_weighted: After adding {} to help prevent underflow, weighted_list is\n".format(\
                             max_elem))
            np.savetxt(debug_file, weighted_list, fmt='%1.4f')
        if debug_level >= 9000:
            # will have been sent masked array
            debug_file.write("Mask of masked array is {}\n".format(ma.getmask(weighted_list)));  
        weighted_list = np.exp(weighted_list)
        # want to zero out any masked entries -- use logical_or with mask of weighted list
        weighted_list = np.logical_or(weighted_list, ma.getmask(weighted_list))*weighted_list
        if (debug_level >= 8000):
            debug_file.write("get_rand_weighted: After converting out of log space, weighted_list is\n")
            np.savetxt(debug_file, weighted_list, fmt='%1.4e')            
    
    # need to create normalized weights if not already - use numpy array
    normalized_weighted_list = array(weighted_list)
    normalized_weighted_list /= normalized_weighted_list.sum(axis=0) # normalize by sum of weights
 
    if debug_level >= 5000:
        debug_file.write("get_rand_weighted: normalized weighted list is \n")
        print_array(normalized_weighted_list, debug_file) 

        
    # generate weighted number
    gen_weight = stats.rv_discrete(values=(np.arange(len(normalized_weighted_list)), \
                                           normalized_weighted_list))
    ret_index = gen_weight.rvs()

    # better way with np.random.choice()
    #gen_weight = np.random.choice(weighted_list, 1, p=weighted_list)
    #ret_index = gen_weight[0]
    
    if debug_level >= 5000:
        debug_file.write("get_rand_weighted: rand index is {}\n".format(ret_index))
           
    return ret_index


# generic function for printing out a hash's keys and values in sorted order
def print_hash(to_print_hash, output_file):
   for hash_k, hash_v in sorted(to_print_hash.items()):
       output_file.write("{}: {}\n".format(hash_k, hash_v))
       output_file.write("")

# generic function for printing out an array's values       
def print_array(to_print_array, output_file):
    for arr_index in range(len(to_print_array)):
        output_file.write("Item {}: {}\n".format(arr_index, to_print_array[arr_index]))                     
                
# useful for being able to define functions lower down than where they're called
# e.g., putting main function first and then function definitions below it
if __name__ == "__main__":
    main()
