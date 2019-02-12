#####
## Copyright (C) 2010-2011, Alexander Rusakov - All Rights Reserved
#####
#put here a class containing data asr for to run tests routine in , f2))
		lines1 = open(f1).readlines() specific build dir
#TODO CMAKE must update some variable
#probably later we found better solution other than hardcoding. 

import subprocess 
import sys
import os
import difflib
import re


class ASRTestcfg:
	asr_sanbox_root_dir = "/home/rusakov/WiseOpt/wiseopt-code/"

	def RootPath(self):
		envval = os.environ['ASR_SANDBOX_DIR'];
		if (envval) :
			self.asr_sanbox_root_dir =  envval + "/";

		return self.asr_sanbox_root_dir
	
				
#utilities for quick tests.
#collecting data to testdir path, execution, comparison vs golden
class ASRTestUtils :	
	#useful for filters. aux
	reg_exclude_string = None
	
 	#compare two files		
	def diffvsgold(self, f1, f2, filter_for_data = None):
		print ("compare file %s vs gold %s " % (f1, f2))
		lines1 = open(f1).readlines()
		lines2 = open(f2).readlines()
		if (filter_for_data) :
			print ("apply filter to data")
			lines1 = filter_for_data(lines1)
			lines2 = filter_for_data(lines2)
		diff=difflib.unified_diff(lines1, lines2)
#		print (list(diff))
#		assert(0)
		return len(list(diff))

	#return 0 in case of succesful run, retval 0 and no diff vs gold
	def runTest(self, binary, args, outfile = [], goldfile = [], filter_for_data = []) : 
		print ("running %s %s in the rundir %s\n" % (binary, args, os.getcwd()))
		cmd = "%s %s" % (binary, args)
		retval = subprocess.call(cmd, shell=True)
		
		#check that outfile exists
		if len(outfile):		
			assert( os.path.isfile(outfile) ) , " expected run output file %s does not exist" % outfile
			
		if retval :
			assert retval == 0 , "exit value != 0"
			return retval
		if len(goldfile) :
			assert( os.path.isfile(goldfile) ) , " gold file %s does not exist" % goldfile
			d = self.diffvsgold(outfile, goldfile, filter_for_data)
			assert d == 0, "outfile %s and %s differ" % (outfile, goldfile)
			if (d) :
				return d;
		
		#success
		return 0

	def setupTestData(self, dst, data_paths) :
		dst.chdir();
		
#		print (" dst %s data_path %s " % (dst, data_paths[0]))
		for src in data_paths :
			retval = subprocess.call("cp -rp %s %s" % (src,dst), shell=True)
			

	#excludes from the list of lines all lines containing an exclude_string
	def filter_lines_by_regexpr(self, lines) :
		r = re.compile(self.reg_exclude_string); 
		rlines = list()
		for l in lines :
			if (r.search(l) == None) :
				rlines.append(l);
		return rlines;				


