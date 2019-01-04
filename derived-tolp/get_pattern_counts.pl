#!/usr/bin/perl -w

# created by Lisa Pearl, 2/28/18

# (1) takes in an input file encoding different properties of verb use within a corpus (formatting described below)
#     --- NOTE: Will ignore any verbs whose linking pattern usages are below a certain threshold (default 5 total usages)
#     --- NOTE: Also takes in a multiplier, which is necessary for extrapolating corpus counts to true counts
# (2) Outputs information related to whether these verbs follow certain linking patterns
#     (a) how many total verbs there are
#     (b) what the Tolerance Principle (TolP) threshold would be for that many verbs
#     (c) how many total verb instances there are 
#     (d) what the total counts are for sem-xxbj in gram-xxbj position, across all verbs
#     (e) how many verbs of the $total_verb_count total don't follow the primary pattern
#     (f) a list of which verbs follow the primary pattern (based on the Tolerance Principle) and which don't
#     (g) the analysis for individual links of the primary pattern:
#         (i) how many total verbs there are with >=threshold uses for link
#         (ii) what the TolP threshold would be for that many verbs
#         (iii) how many verbs of the $total_link_verb_count don't follow that link
#         (iv) a list of which verbs follow that primary link (based on the Tolerance Principle) and which don't


## input format
## (such as for brown-eve-valian.AbsCl-baker.thresh5.stats or any other file created to be used as input to the linking pattern predicate-learner python code at https://github.com/lisapearl/linking-problem-code)
# verbs are listed out in capitals with accompanying information following
#ANSWER
#anim-gram-subj vs. inanim: 7 1
#anim-gram-obj vs. inanim: 0 8
#anim-gram-iobj vs. inanim: 0 0
#sem-subj as gram-subj vs. gram-obj vs. gram-iobj: 7 0 0
#sem-obj as gram-subj vs. gram-obj vs. gram-iobj: 1 8 0
#sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: 0 0 0
#frames:
#NP VB: 1
#...
#***
#ASK
#...


## Output (full file)
# Total verbs: $total_verb_count
# TolP verb threshold: $tolp_threshold
# Verbs not following primary pattern: $not_primary_verbs
# Total verb usage instances: $total_instances
# Counts for total verb usage instances:
# sem-subj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
# sem-obj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
# sem-iobj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
#
# verbs not following the primary pattern:
# $not_primary_verb1
# $not_primary_verb2
# ...
#
# verbs following the primary pattern:
# $primary_verb1
# $primary_verb2
# ...
# ************
# Link: sem-subj as gram-subj
#
# Total link verbs >= threshold: $total_link_verb_count
# TolP verb threshold: $tolp_link_threshold
# Verb count not following primary link: $not_primary_link_verb_count
# Verbs not following primary link: $not_primary_link_verbs
# Verbs following the primary link: $primary_link_verbs
#
# ***********
# Link: gram-subj as sem-subj
# ....

# (will also include sem-obj as gram-obj, gram-obj as sem-obj, semi-iobj as gram-iobj, and gram-iobj as semi-iobj)


## Example Usage ##
# get_pattern_counts.pl --input_file 'path/to/input_file' --output_file 'path/to/output/file' --threshold 2 --multiplier 60.2


###################

$output_file = "pattern.counts"; # default
$threshold = 5; # default
$multiplier = 1; # default is to leave it alone

# Get input folder (required) and output file (if specified)
use Getopt::Long;
GetOptions("input_file=s" => \$input_file, # file where verb info is 
	   "output_file:s" => \$output_file, # can specify output file path, default = pattern.counts in current directory
	   "threshold:i" => \$threshold, # minimum linking pattern uses to be included in calculation
	   "multiplier:f" => \$multiplier # multiplier to extrapolate from corpus counts to true counts
	  );

print(STDERR "Reading verb info from input file $input_file\n");
print(STDERR "Printing out pattern count info to output file $output_file\n");
print(STDERR "Only considering verbs with at least $threshold linking pattern uses\n");
print(STDERR "Multiplier for extrapolating corpus counts to true counts = $multiplier\n");

### Data structure verb_info
%verbs_info = ();
# $verbs_info->{"total_verb_count"} (total verb count)
$verbs_info{"total_verb_count"} = 0;
# $verbs_info->{"tolp_threshold"} (tolp threshold for total verb count)
$verbs_info{"tolp_threshold"} = 0.0;
# $verbs_info->{"not_primary_verb_count"} (not primary verb count)
$verbs_info{"not_primary_verb_count"} = 0;
# $verbs_info->{"total_instances"} (verb usage instances)
$verbs_info{"total_instances"} = 0;
# $verbs_info->{"instances"}->
#             {"sem-subj"}->
#                           {"gram-subj"}, {"gram-obj"}, {"gram-iobj"}
#             {"sem-obj"}->
#                           ...
#             {"sem-iobj"}->
# ...
### initialize these
$verbs_info{"instances"}{"sem-subj"}{"gram-subj"} = 0;
$verbs_info{"instances"}{"sem-subj"}{"gram-obj"} = 0;
$verbs_info{"instances"}{"sem-subj"}{"gram-iobj"} = 0;
$verbs_info{"instances"}{"sem-obj"}{"gram-subj"} = 0;
$verbs_info{"instances"}{"sem-obj"}{"gram-obj"} = 0;
$verbs_info{"instances"}{"sem-obj"}{"gram-iobj"} = 0;
$verbs_info{"instances"}{"sem-iobj"}{"gram-subj"} = 0;
$verbs_info{"instances"}{"sem-iobj"}{"gram-obj"} = 0;
$verbs_info{"instances"}{"sem-iobj"}{"gram-iobj"} = 0;

# verbs_info{"links"}{$first} = ...info associated with $first as $second (e.g., sem-subj as gram-subj)
#            ...{"total_link_verb_count"} = # of verbs with >= threshold uses of $first
#            ...{"link_tolp_threshold"} = TolP threshold, based on total_link_verb_count
#            ...{"not_primary_link_verb_count"} = # of verbs that aren't primary link verbs
## will need to be initialized for each link we do

# $verbs_info->{"verbs"}->
#             {$verb}->{"instances"}->{"sem-subj"}->...
#                                   ->{"sem-obj"}->...
#                                   ->{"sem-iobj"}->...
#                    ->{"total_instances"}
#                    ->{"not_primary_instances"}
#                    ->{"tolp_threshold"}
#                    ->{"not_primary_verb"} = 1 or 0
#                    ->{"links"}{$first} = .... info associated with each link of $first to $second
#                              {"above_threshold"} = verb has enough instances along this link to be included
#                              {"total_link_instances"}
#                              {"not_primary_link_instances"}
#                              {"link_tolp_threshold"}
#                              {"not_primary_link_verb"} = 1 or 0

# "primary_no_verbs" = array of verbs not following the primary pattern
@primary_no_verbs = ();
# "primary_yes_verbs" = array of verbs following the primary pattern
@primary_yes_verbs = ();


## arrays of verbs for each link: in hash of arrayrefs
%link_verbs = ();
# $link_verbs{$first}{"no"} = arrayref of verbs not obeying the link
# $link_verbs{$first}{"yes"} = arrayref of verbs obeying the link



# (1) Read in verb information from input file and populate parts of verb_info data structure, given multiplier
read_in_verb_info();
# delete verbs with linking pattern uses below $threshold * $multiplier
delete_below_threshold_verbs();
print("debug, after reading verbs in from $input_file, verb info is this:\n");
print_verb_info();
# (2) Calculate remaining components of verb_info data structure
calculate_verb_info();
# (3) Calculate same information for individual links
calculate_link_info();
# (4) Print results to output_file
print_results();

####################
sub calculate_link_info{

  # for each of the six primary pattern links
  # sem-subj to gram-subj
  calculate_specific_link("sem-subj","gram-subj");
  # gram-subj to sem-subj
  calculate_specific_link("gram-subj", "sem-subj");
  # sem-obj to gram-obj
  calculate_specific_link("sem-obj","gram-obj");
  # gram-obj to sem-obj
  calculate_specific_link("gram-obj", "sem-obj");
  # sem-iobj to gram-iobj
  calculate_specific_link("sem-iobj","gram-iobj");
  # gram-iobj to sem-iobj
  calculate_specific_link("gram-iobj", "sem-iobj");
}

# populate all the info for a specific link like sem-subj as gram-subj or gram-subj as sem-subj
sub calculate_specific_link{
  my ($first, $second) = @_;
  print("debug, calculate_specific_link: link is $first to $second\n"); 

  # initialize total verbs for this link
  # verbs_info{"links"}
  #            ...{"total_link_verb_count"} = # of verbs with >= threshold uses of $first
  $verbs_info{"links"}{$first}{"total_link_verb_count"} = 0;
  #            ...{"not_primary_link_verb_count"} = # of verbs that aren't primary link verbs
  $verbs_info{"links"}{$first}{"not_primary_link_verb_count"} = 0;
  # initialize link_verbs{$first} = empty arrayref
  $link_verbs{$first}{"yes"} = ();
  $link_verbs{$first}{"no"} = ();
  
  # for each verb...
  foreach my $verb (sort(keys($verbs_info{"verbs"}))){
    #print("debug, calculate_specific_link: getting link info for $verb\n");
    # $verbs_info->{"verbs"}->
    #             {$verb}->{"links"}{$first} = .... info associated with each link of $first to $second
    #                              {"above_threshold"} = verb has enough instances along this link to be included
    #                              {"total_link_instances"}
    #                              {"not_primary_link_instances"}
    #                              {"link_tolp_threshold"}
    #                              {"not_primary_link_verb"} = 1 or 0

    
    # total_link_instances: sum across row if $first is sem-xxx, or down column is $first is gram-xxx
    $verbs_info{"verbs"}{$verb}{"links"}{$first}{"total_link_instances"} = get_link_total_instances($verb, $first);
    #print("debug, calculate_specific_link: total link instances of $first is verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"total_link_instances\"} = $verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"total_link_instances\"}\n");

    # above_threshold and total_link_verb_count
    # update above_threshold and verbs_info{"links"}{"total_link_verb_count"}
    # verbs_info{"links"}
    #            ...{"total_link_verb_count"} = # of verbs with >= threshold uses of $first
    if($verbs_info{"verbs"}{$verb}{"links"}{$first}{"total_link_instances"} >= $threshold){
      #print("debug, calculate_specific_link: total above threshold $threshold\n");
      $verbs_info{"verbs"}{$verb}{"links"}{$first}{"above_threshold"} = 1;
      #print("debug, calculcate_specific_link: before updating verbs_info{\"links\"}{$first}{\"total_link_verb_count\"} = $verbs_info{\"links\"}{$first}{\"total_link_verb_count\"}\n");
      $verbs_info{"links"}{$first}{"total_link_verb_count"} += 1;
      #print("debug, calculcate_specific_link: verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"above_threshold\"} = $verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"above_threshold\"}\n");
      #print("debug, calculate_specific_link: verbs_info{\"links\"}{$first}{\"total_link_verb_count\"} = $verbs_info{\"links\"}{$first}{\"total_link_verb_count\"}\n");

      #not_primary_link_instances: those that don't match the suffix (subj, obj, or iobj)
      $verbs_info{"verbs"}{$verb}{"links"}{$first}{"not_primary_link_instances"} = get_not_primary_link_instances($verb, $first);
      #print("debug, calculate_specific_link: verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"not_primary_link_instances\"} = $verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"not_primary_link_instances\"}\n");


      # link_tolp_threshold, based on total_link_instances
      # only calculate if above_threshold
      # N/log(N) where N = total_link_instances
      my $verb_n = $verbs_info{"verbs"}{$verb}{"links"}{$first}{"total_link_instances"};
      #print("debug, calculate_specific_links: n for $verb 's tolp calculation = $verb_n\n");
      $verbs_info{"verbs"}{$verb}{"links"}{$first}{"link_tolp_threshold"} = $verb_n/log($verb_n);
      #print("debug, calculate_specific_link: verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"link_tolp_threshold\"} = $verb_n/log($verb_n) = $verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"link_tolp_threshold\"}\n");

      ####################
      #not_primary_link_verb = not_primary_link_instances < link_tolp_threshold
      # only calculate if above_threshold
      if($verbs_info{"verbs"}{$verb}{"links"}{$first}{"not_primary_link_instances"} <
	 $verbs_info{"verbs"}{$verb}{"links"}{$first}{"link_tolp_threshold"}){
	#print("debug, calculate_specific_links: not_primary exceptions less than threshold, so $verb is primary link verb \n");
	## update "not_primary_link_verb" field
	$verbs_info{"verbs"}{$verb}{"links"}{$first}{"not_primary_link_verb"} = 0;

	## update arrays for specific links
	# $link_verbs{$first}{"yes"} = arrayref of verbs obeying the link
	push(@{$link_verbs{$first}{"yes"}}, $verb);
	#print("debug, calculate_specific_link: yes_verbs is now ");
	#foreach my $vb (sort @{$link_verbs{$first}{"yes"}}){
	#  print("$vb ");
	#}
	#print("\n");
		
      }else{
	## not_primary_link verb
	$verbs_info{"verbs"}{$verb}{"links"}{$first}{"not_primary_link_verb"} = 1;
	## update arrays for specific links
	# $link_verbs{$first}{"no"} = arrayref of verbs obeying the link
	push(@{$link_verbs{$first}{"no"}}, $verb);
	#print("debug, calculate_specific_link: no_verbs is now ");
	#foreach my $vb (sort @{$link_verbs{$first}{"no"}}){
	#  print("$vb ");
	#}
	#print("\n");
      }

      ## update $verbs_info{"links"}{$first}{"not_primary_link_verb_count"}
      $verbs_info{"links"}{$first}{"not_primary_link_verb_count"} += 1;
      #print("debug, calculate_specific_link: verbs_info{\"links\"}{$first}{\"not_primary_link_verb_count\"} = $verbs_info{\"links\"}{$first}{\"not_primary_link_verb_count\"}\n");
      
    }else{
      #ignore for purposes of this particular link
      print("debug, calculate_specific_link: ***ignoring $verb because too few link instances\n");
      $verbs_info{"verbs"}{$verb}{"links"}{$first}{"above_threshold"} = 0;
      #print("debug, calculcate_specific_link: verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"above_threshold\"} = $verbs_info{\"verbs\"}{$verb}{\"links\"}{$first}{\"above_threshold\"}\n");
    }
  }

  # verbs_info{"links"}{$first}  
  #            ...{"link_tolp_threshold"} = TolP threshold, based on total_link_verb_count
  my $total_v = $verbs_info{"links"}{$first}{"total_link_verb_count"};
  $verbs_info{"links"}{$first}{"link_tolp_threshold"} = $total_v/log($total_v);
  print("debug, calculate_specific_link: verbs_info{\"links\"}{$first}{\"link_tolp_threshold\"} for $total_v verbs with above_threshold link instances = $verbs_info{\"links\"}{$first}{\"link_tolp_threshold\"}\n");
 
}


sub get_not_primary_link_instances{
  my ($verb, $first) = @_;

  my $not_primary_total = 0; 
  #print("debug, get_not_primary_link_instances: total_instances = $not_primary_total\n");
  
  #those that don't match the suffix (subj, obj, or iobj)
  # if first is sem-xxx, sum the {$first}{gram-xxx} that don't match
  if($first eq "sem-subj"){
    #print("debug, get_not_primary_link_instances: not_primary_total = $not_primary_total + $verbs_info{\"verbs\"}{$verb}{\"instances\"}{$first}{\"gram-obj\"} + $verbs_info{\"verbs\"}{$verb}{\"instances\"}{$first}{\"gram-iobj\"}\n");
    $not_primary_total +=
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-obj"} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-iobj"};
    #print("debug, get_not_primary_link_instances: not_primary_total = $not_primary_total\n");
  }elsif($first eq "sem-obj"){
    $not_primary_total +=
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-subj"} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-iobj"};
    
  }elsif($first eq "sem-iobj"){
        $not_primary_total +=
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-subj"} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-obj"};

  }
  # if first is gram-xxx, sum the {sem-xxx}{$first} that don't match
  elsif($first eq "gram-subj"){
    $not_primary_total += 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-obj"}{$first} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-iobj"}{$first};
    #print("debug, get_not_primary_link_instances: not_primary_total = $not_primary_total\n");    
  }elsif($first eq "gram-obj"){
    $not_primary_total += 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-subj"}{$first} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-iobj"}{$first};
    
  }elsif($first eq "gram-iobj"){
    $not_primary_total += 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-subj"}{$first} + 
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-obj"}{$first};   
  }

  return $not_primary_total;
}

sub get_link_total_instances{
  my ($verb, $first) = @_;
  my $total = 0;

  #print("debug, get_link_total_instances: instances from $first for $verb\n");
  
  if($first=~/^sem/){
    #print("debug, calculate_specific_link: $first is sem-xxx so sum across {sem-xxx}{...} entries\n");
    #print("debug, calculate_specific_link: verbs_info{\"verbs\"}{$verb}{\"instances\"}{$first}{\"gram-subj\"} = $verbs_info{\"verbs\"}{$verb}{\"instances\"}{$first}{\"gram-subj\"}\n");
    $total = 
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-subj"} +
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-obj"} +
      $verbs_info{"verbs"}{$verb}{"instances"}{$first}{"gram-iobj"};   
  }elsif($first=~/^gram/){
    #print("debug, calculate_specific_link: $first is gram-xxx so sum across {...}{$first} entries\n");
        #print("debug, calculate_specific_link: verbs_info{\"verbs\"}{$verb}{\"instances\"}{\"sem-subj\"}{\"gram-subj\"} = $verbs_info{\"verbs\"}{$verb}{\"instances\"}{\"sem-subj\"}{\"gram-subj\"}\n");
    #print("debug, calculate_specific_link: verbs_info{\"verbs\"}{$verb}{\"instances\"}{\"sem-subj\"}{\"$first\"} = $verbs_info{\"verbs\"}{$verb}{\"instances\"}{\"sem-subj\"}{\"$first\"}\n"); 
    $total =
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-subj"}{$first} +
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-obj"}{$first} +
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-iobj"}{$first}; 
    
  }else{
    #print("debug, calculate_specific_link: first link not recognized ($first)\n");
  }

  return $total;
}
  
## Calculate remaining components of verb_info data structure
## Updated to include individual link calculations
sub calculate_verb_info{

  # total verb count = number of keys in verb_info{"verbs"}
  my $total_verb_count = scalar(keys($verbs_info{"verbs"}));
  $verbs_info{"total_verb_count"} = $total_verb_count;
  print("debug, calculate_verb_info: total verbs = $verbs_info{\"total_verb_count\"}\n");
  
  # tolp_threshold = N/ln N
  my $tolp_threshold = $total_verb_count/log($total_verb_count);
  $verbs_info{"tolp_threshold"} = $tolp_threshold;
  print("debug, calculate_verb_info: tolp_threshold = $verbs_info{\"tolp_threshold\"}\n");

  # total_instances = sum of entries in verb_info{"instances"}
  my $total_instances = 0;
  foreach my $sem (keys($verbs_info{"instances"})){
    foreach my $gram (keys($verbs_info{"instances"}{$sem})){
      $total_instances += $verbs_info{"instances"}{$sem}{$gram};
    }
  }
  $verbs_info{"total_instances"} = $total_instances;
  print("debug, calculate_verb_info: total_instances = $verbs_info{\"total_instances\"}\n");

  # now process all the verbs individually so we can populate
  # not_primary_verb_count (this is based on whether an individual verb has too many not_primary instances),
  #    verb total_instances, verb not_primary_instances, verb tolp_threshold,
  #    verb not_primary_verb (yes or no) -- this feeds into not_primary_verb_count and the arrays below
  # primary_no_verbs
  # primary_yes_verbs
  foreach my $verb (sort(keys($verbs_info{"verbs"}))){
    my $vb_total_instances = $verbs_info{"verbs"}{$verb}{"total_instances"};
    
    # not_primary_instances
    my $vb_not_primary_instances = $vb_total_instances -
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-subj"}{"gram-subj"} -
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-obj"}{"gram-obj"} -
      $verbs_info{"verbs"}{$verb}{"instances"}{"sem-iobj"}{"gram-iobj"};
    $verbs_info{"verbs"}{$verb}{"not_primary_instances"} = $vb_not_primary_instances;
    print("debug, calculate_verb_info: $verb not_primary_instances = $verbs_info{\"verbs\"}{$verb}{\"not_primary_instances\"}\n");
      
    # tolp_threshold
    my $vb_tolp_threshold = $vb_total_instances/log($vb_total_instances);
    $verbs_info{"verbs"}{$verb}{"tolp_threshold"} = $vb_tolp_threshold;
    print("debug, calculate_verb_info: $verb tolp_threshold = $verbs_info{\"verbs\"}{$verb}{\"tolp_threshold\"}\n");

    # not_primary_verb: if vb_not_primary_instances > $vb_tolp_threshold
    # update primary_no and primary_yes verb arrays
    if($vb_not_primary_instances > $vb_tolp_threshold){
      $verbs_info{"verbs"}{$verb}{"not_primary_verb"} = "yes";
      push(@primary_no_verbs, $verb);
      #print("debug, calculate_verb_info: primary_no_verbs now @primary_no_verbs\n");     
    }else{
      $verbs_info{"verbs"}{$verb}{"not_primary_verb"} = "no";
      push(@primary_yes_verbs, $verb);
      #print("debug, calculate_verb_info: primary_yes_verbs now @primary_yes_verbs\n");  
    }
    print("debug, calculate_verb_info: $verb not_primary_verb = $verbs_info{\"verbs\"}{$verb}{\"not_primary_verb\"}\n");
  }

  # now do not_primary_verb_count
  $verbs_info{"not_primary_verb_count"} = scalar(@primary_no_verbs);
  print("debug, calculate_verb_info: \# of not primary verbs = $verbs_info{\"not_primary_verb_count\"}\n");
}

sub delete_below_threshold_verbs{
  # if total linking pattern uses below $threshold * $multiplier, delete $verb element
  foreach my $verb (sort(keys($verbs_info{"verbs"}))){
    my $vb_total_instances = 0;
    foreach my $sem (keys($verbs_info{"verbs"}{$verb}{"instances"})){
      foreach my $gram (keys($verbs_info{"verbs"}{$verb}{"instances"}{$sem})){
	$vb_total_instances += $verbs_info{"verbs"}{$verb}{"instances"}{$sem}{$gram};
      }
    }

    # delete if below $threshold, otherwise add to verbs_info
    if ($vb_total_instances < $threshold * $multiplier){
      print("debug, delete_below_threshold_verbs: ***deleting $verb with $vb_total_instances uses\n");
      delete($verbs_info{"verbs"}{$verb});
    }else{
      $verbs_info{"verbs"}{$verb}{"total_instances"} = $vb_total_instances;
      print("debug, delete_below_threshold_verbs: $verb total_instances = $verbs_info{\"verbs\"}{$verb}{\"total_instances\"}\n");
    }
  }

}


# Read in verb information from input file and populate relevant bits of verb_info, given multiplier
sub read_in_verb_info{

  my $verb; # for holding verb being processed
  # open input_file
  open(VERBS_IN, "$input_file") || die("Couldn't open $input_file to read in verb info\n");

  # read in verb info
  # format
  #ANSWER
  #anim-gram-subj vs. inanim: 7 1
  #anim-gram-obj vs. inanim: 0 8
  #anim-gram-iobj vs. inanim: 0 0
  #sem-subj as gram-subj vs. gram-obj vs. gram-iobj: 7 0 0
  #sem-obj as gram-subj vs. gram-obj vs. gram-iobj: 1 8 0
  #sem-iobj as gram-subj vs. gram-obj vs. gram-iobj: 0 0 0
  #frames:
  #NP VB: 1
  #...
  #***

  while(defined($verbline = <VERBS_IN>)){
    #print("debug, read_in_verb_info: verbline is $verbline\n");
    if($verbline =~ /^([A-Z][^\s]+)\n/){
      $verb = $1;
      #print("debug, read_in_verb_info: current verb is $verb\n");
      # then skip over animacy info (next three lines)
      $verbline = <VERBS_IN>;
      $verbline = <VERBS_IN>;
      $verbline = <VERBS_IN>;
      # get sem-xxbj as gram-xxbj lines
      $verbline = <VERBS_IN>;
      #print("debug, read_in_verb_info: verbline is $verbline\n");
      if($verbline =~ /^sem-subj as gram.*: (\d+) (\d+) (\d+)/){
	#print("debug, read_in_verb_info: sem-subj as gram-subj: $1, obj: $2, iobj: $3\n");
	# populate for sem-subj as gram-xxbj
	populate_verbs_info($verb,"sem-subj", $1, $2, $3);
	#print("debug, read_in_verb_info: verbs_info after reading in sem-xxbj as gram-xxbj info for $verb\n");
	#print_verb_info();

	# get sem-obj line
	$verbline = <VERBS_IN>;
	#print("debug, read_in_verb_info: verbline is $verbline\n");
	if($verbline =~ /^sem-obj as gram.*: (\d+) (\d+) (\d+)/){
	  populate_verbs_info($verb,"sem-obj", $1, $2, $3);
	  #print("debug, read_in_verb_info: verbs_info after reading in sem-xxbj as gram-xxbj info for $verb\n");
	  #print_verb_info();

	  #get sem-iobj line
	  $verbline = <VERBS_IN>;
	  #print("debug, read_in_verb_info: verbline is $verbline\n");
	  if($verbline =~ /^sem-iobj as gram.*: (\d+) (\d+) (\d+)/){
	    populate_verbs_info($verb,"sem-iobj", $1, $2, $3);
	    #print("debug, read_in_verb_info: verbs_info after reading in sem-xxbj as gram-xxbj info for $verb\n");
	    #print_verb_info();
	  }	  
	}
      }
     # now just churn through the rest of the lines till we hit the next verb
    }
  }
  
  close(VERBS_IN);
}

# incorporate multiplier effect here
sub populate_verbs_info{
  my ($verb, $sem, $g_subj, $g_obj, $g_iobj) = @_;

  #print("debug, populate_verbs_info: populating for $verb and sem val $sem with gram-subj $g_subj, gram-obj $g_obj, and gram-iobj $g_iobj vals\n");
	
  # populate info for this specific $verb
  $verbs_info{"verbs"}{$verb}{"instances"}{$sem}{"gram-subj"} = $g_subj * $multiplier;
  $verbs_info{"verbs"}{$verb}{"instances"}{$sem}{"gram-obj"} = $g_obj * $multiplier;
  $verbs_info{"verbs"}{$verb}{"instances"}{$sem}{"gram-iobj"} = $g_iobj * $multiplier;

  #print("debug, populate_verbs_info info for $sem\n");
  #print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$sem}{\"gram-subj\"} ");
  #print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$sem}{\"gram-obj\"} ");
  #print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$sem}{\"gram-iobj\"}\n");
  #print("and verbs info now\n");
  #print_verb_info();
  
  # update info for general verbs collective
  $verbs_info{"instances"}{$sem}{"gram-subj"} += $g_subj * $multiplier;
  $verbs_info{"instances"}{$sem}{"gram-obj"} += $g_obj * $multiplier;
  $verbs_info{"instances"}{$sem}{"gram-iobj"} += $g_iobj * $multiplier;

  #print("debug, populate_verbs_info\n");
  #print("all verbs gram-subj: $verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-subj\"}\n");
  #print("all verbs gram-obj: $verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-obj\"}\n");
  #print("all verbs gram-iobj: $verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-iobj\"}\n");

  #print("debug, populate_verbs_info: after populating for $sem\n");
  #print_verb_info();

}

sub print_verb_info{

  # print specific verb instance info
  foreach my $verb (sort(keys($verbs_info{"verbs"}))){
    print("Verb instance information for $verb\n");
    foreach my $semval (sort(keys(%{$verbs_info{"verbs"}{$verb}{"instances"}}))){
      print("$semval as gram-subj, obj, iobj: ");
      print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$semval}{\"gram-subj\"} ");
      print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$semval}{\"gram-obj\"} ");      
      print("$verbs_info{\"verbs\"}{$verb}{\"instances\"}{$semval}{\"gram-iobj\"}\n");
    }
  }

  # all verb instances
  print("All verb instances\n");
  foreach my $semval (sort(keys(%{$verbs_info{"instances"}}))){
    print("$semval as gram-subj vs. gram-obj vs. gram-iobj: ");
    print("$verbs_info{\"instances\"}{$semval}{\"gram-subj\"} ");
    print("$verbs_info{\"instances\"}{$semval}{\"gram-obj\"} ");
    print("$verbs_info{\"instances\"}{$semval}{\"gram-iobj\"}\n");
  }
  
}

sub print_results{
  ## Output (full file)
  # Total verbs: $total_verb_count
  # TolP verb threshold: $tolp_threshold
  # Verbs not following primary pattern: $not_primary_verbs
  # Total verb usage instances: $total_instances
  # Counts for total verb usage instances:
  # sem-subj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  # sem-obj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  # sem-iobj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  #
  # verbs not following the primary pattern:
  # $not_primary_verb1
  # $not_primary_verb2
  # ...
  #
  # verbs following the primary pattern:
  # $primary_verb1
  # $primary_verb2
  # ...
  # ************
  # Link: sem-subj as gram-subj
  #
  # Total link verbs >= threshold: $total_link_verb_count
  # TolP verb threshold: $tolp_link_threshold
  # Verb count not following primary link: $not_primary_link_verb_count
  # Verbs not following primary link: $not_primary_link_verbs
  # Verbs following the primary link: $primary_link_verbs
  #
  # ***********
  # Link: gram-subj as sem-subj
  # ....
  
  # (will also include sem-obj as gram-obj, gram-obj as sem-obj, semi-iobj as gram-iobj, and gram-iobj as semi-iobj)


  # open output file
  open(OUT, ">$output_file") || die("Couldn't open $output_file to print out results\n");
  print(OUT "Ignoring verbs with fewer than $threshold linking pattern uses\n");
  # Total verbs: $total_verb_count
  print(OUT "Total verbs: $verbs_info{\"total_verb_count\"}\n");
  # TolP verb threshold: $tolp_threshold
  print(OUT "TolP verb threshold: $verbs_info{\"tolp_threshold\"}\n");
  # Verbs not following primary pattern: $not_primary_verbs
  print(OUT "Verbs not following primary pattern: $verbs_info{\"not_primary_verb_count\"}\n");
  # Total verb usage instances: $total_instances
  print(OUT "Total verb usage instances: $verbs_info{\"total_instances\"}\n");
  # Counts for total verb usage instances:
  # sem-subj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  # sem-obj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  # sem-iobj as gram-subj vs. gram-obj. vs. gram-iobj: \d+ \d+ \d+
  print(OUT "Counts for total verb usage instances:\n");
  # sem-subj
  print(OUT "sem-subj as gram-subj vs. gram-obj. vs. gram-iobj: ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-subj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-obj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-subj\"}{\"gram-iobj\"}\n");
  # sem-obj
  print(OUT "sem-obj as gram-subj vs. gram-obj. vs. gram-iobj: ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-obj\"}{\"gram-subj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-obj\"}{\"gram-obj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-obj\"}{\"gram-iobj\"}\n");
  # sem-iobj
  print(OUT "sem-iobj as gram-subj vs. gram-obj. vs. gram-iobj: ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-iobj\"}{\"gram-subj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-iobj\"}{\"gram-obj\"} ");
  print(OUT "$verbs_info{\"instances\"}{\"sem-iobj\"}{\"gram-iobj\"}\n");
  #
  # verbs not following the primary pattern:
  print(OUT "\nverbs not following the primary pattern\n");
  foreach my $verb (@primary_no_verbs){
    # $not_primary_verb1
    # $not_primary_verb2
    # ...
    print(OUT "$verb ");
  }
  print(OUT "\n");
  #
  # verbs following the primary pattern:
  print(OUT "\nverbs following the primary pattern:\n");
  foreach my $verb (@primary_yes_verbs){
    # $primary_verb1
    # $primary_verb2
    # ...
    print(OUT "$verb ");
  }
  print(OUT "\n");

  ### each of the links (TO DO: turn this into a function call?)
  # Link: sem-subj as gram-subj
  print(OUT "**********\nLink: sem-subj as gram-subj\n\n");
  print_link("sem-subj");

  # Link: gram-subj as sem-subj
  print(OUT "**********\nLink: gram-subj as sem-subj\n\n");
  print_link("gram-subj");  

  # Link: sem-obj as gram-obj
  print(OUT "**********\nLink: sem-obj as gram-obj\n\n");
  print_link("sem-obj");

  # Link: gram-obj as sem-obj
  print(OUT "**********\nLink: gram-obj as sem-obj\n\n");
  print_link("gram-obj");  

  # Link: sem-iobj as gram-iobj
  print(OUT "**********\nLink: sem-iobj as gram-iobj\n\n");
  print_link("sem-iobj");

  # Link: gram-iobj as sem-iobj
  print(OUT "**********\nLink: gram-iobj as sem-iobj\n\n");
  print_link("gram-iobj");  
 
  
  close(OUT);

}

sub print_link{
  my ($link) = @_;

  # Total link verbs >= threshold: $total_link_verb_count
  print(OUT "Total link verbs >= $threshold: $verbs_info{\"links\"}{$link}{\"total_link_verb_count\"}\n");
  # TolP verb threshold: $tolp_link_threshold
  print(OUT "TolP verb threshold: $verbs_info{\"links\"}{$link}{\"link_tolp_threshold\"}\n");
  # Verb count not following primary link: $not_primary_link_verb_count
  print(OUT "Verb count not following primary link: ");
  my $not_vb_count = scalar(@{$link_verbs{$link}{"no"}});
  print(OUT "$not_vb_count\n");
  # Verbs not following primary link: $not_primary_link_verbs
  print(OUT "\nVerbs not following primary link: @{$link_verbs{$link}{\"no\"}}\n");
  # Verbs following the primary link: $primary_link_verbs
  print(OUT "\nVerbs following the primary link: @{$link_verbs{$link}{\"yes\"}}\n");

  
}


=begin comment

Where to put multi-line comments

=end comment

=cut
