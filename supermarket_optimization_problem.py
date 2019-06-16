'''
Use pyspark and set the below variables:
 export PYSPARK_PYTHON=python3 
 export PYSPARK_DRIVER_PYTHON=python3
 
 Get the infrequent sets of pairs of trans which less than support sigma
 filter out such pairs and ignore. 
 Get the most frequent N+1 pairs in recursive way using previous N pairs
 Run as:  python supermarket_optimization_problem.py -i retail_25k.dat -s 4 -o result
 python is alias for python3.5 in my environment
'''

from pyspark.sql.session import SparkSession
from pyspark import SparkContext

import argparse
import datetime
from itertools import combinations


def get_time():
    return str(datetime.datetime.now())[:19]

def transact_rows(line, infreq_skus):
    line = line.strip().split(' ')
    line = [sku for sku in line if sku not in infreq_skus]
    return line


def get_maxitem(dataFrame):
    return dataFrame.map(lambda line: len(line)).max()


def save_result(results, file_path):
    result_string = []
    for result in results:
        # determine the N for the current result list
        N = len(result[0][0].split(','))
        for r in result:
            # format result as: <item set (N)>, <co-occurrence frequency>, <item 1 id >, <item 2 id>, ..., <item N id>
            result_string.append('%i,%i,%s' %(N, r[1], r[0]))

    with open(file_path, 'w') as f:
        f.write('\n'.join(result_string))
    print('%s: Save results into %s' %(get_time(), file_path))


def get_lessfreq_skus(dataFrame, sigma):
    transactions = dataFrame.collect()
    sku_frq = {}
    for line in transactions:
        skus = line.strip().split(' ')
        for sku in skus:
            if sku in sku_frq.keys():
                sku_frq[sku] += 1
            else:
                sku_frq[sku] = 1
    # gen a dic of skus less than support
    infreq_skus = []
    for s in sku_frq.keys():
        if sku_frq[s] < sigma:
            infreq_skus.append(s)

    return infreq_skus


def get_combs(sku_list, N):
    # string key for each combo as 'sku1,sku2,sku3,...'
    return [','.join(c) for c in combinations(sku_list, N)]


def prev_results_to_new_combinations(sku_set, prev_results):
    combs = set()  
    for result_prev in prev_results:
        # create N+1 possible pairs including 
        # previous N skus with new sku in current transaction
        if result_prev.issubset(sku_set):      
            for sku in sku_set - result_prev:
                combs.add(','.join(sorted(result_prev.union(set([sku])))))
    return list(combs)


def get_original_freq_pairs(dataFrame, size, sigma):
    # map: map each comb into (comb, 1) k/v
    comb_count_dataFrame = dataFrame.flatMap(lambda sku_list: get_combs(sku_list, size)) \
                     .map(lambda c: (c,1)) \
                     .reduceByKey(lambda a, b: a + b)

    # eliminate pairs with freq less than support
    results = comb_count_dataFrame.filter(lambda comb_count: comb_count[1]>=sigma).collect()
    return results
    
def freq_size_pairs(dataFrame, sigma, prev_results):
    comb_count_dataFrame = dataFrame.flatMap(lambda sku_list: prev_results_to_new_combinations(set(sku_list), prev_results)).map(lambda c: (c,1)).reduceByKey(lambda a, b: a + b)
    results = comb_count_dataFrame.filter(lambda comb_count: comb_count[1]>=sigma).collect()
    return results


def find_frequent_pairs(dataFrame, sigma):
    min_size = 3
    max_size = get_maxitem(dataFrame)
    results = []
    
    for size in range(min_size, max_size+1):
        print('%s: Process pairs with size  = %i ' %(get_time(), size))
        if size == min_size:
            new_results = get_original_freq_pairs(dataFrame, min_size, sigma)
        else:
            new_results = freq_size_pairs(dataFrame, sigma, prev_results)
        print('%s: Got %i pairs' %(get_time(),len(new_results)))

        # break from size loop if there is no size pairs
        if len(new_results) == 0:
            print('%s: if no response, break!' %get_time())
            break
            
        # add to final results
        results.append(new_results)
        # Get prev_results in format we want!
        prev_results = [set(r[0].split(',')) for r in new_results]
 
    return results


if __name__ == '__main__':
    # Build an arg parser 
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sigma", required=True, help = "sigma - min threshold")
    ap.add_argument("-i", "--input", required=True, help = "input file")
    ap.add_argument("-o", "--result", required=True, help = "result file")
    args = vars(ap.parse_args())

    # Create a Spark session
    sess = SparkSession.builder.getOrCreate()
    # Get the dataFrame
    data = sess.sparkContext.textFile(args['input'])
    # Get infrequent product skus
    infreq_skus = get_lessfreq_skus(data, int(args['sigma']))
    data = data.map(lambda line: transact_rows(line, infreq_skus)).cache()
    # Get frequent bought pairs
    results = find_frequent_pairs(data, int(args['sigma']))
    # Save the result
    save_result(results, args['result'])

