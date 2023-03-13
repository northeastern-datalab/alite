# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:32:26 2021

@author: khati
"""
import glob
import pandas as pd
import numpy as np
import os
import time
import cProfile
import pstats
from operator import itemgetter
import sys
import alite_fd as cfd
import BiconnectedComponents as bcc
import strongly_connected_components as scc

#preprocess input tables
def preprocess(table):
    table = table.drop_duplicates().reset_index(drop=True)
    table = table.replace(r'^\s*$',np.nan, regex=True)
    table = table.replace("-",np.nan)
    table = table.replace(r"\N",np.nan)
    table.columns = map(str.lower, table.columns)
    #table = table.replace(r'^\s*$',"undefinedval", regex=True) #convert inherit nulls to "undefinedval"
    #table = table.replace(np.nan,"undefinedval", regex=True) #convert inherit nulls to "undefinedval"
    table = table.applymap(str) 
    table = table.apply(lambda x: x.str.lower()) #convert to lower case
    table = table.apply(lambda x: x.str.strip()) #strip leading and trailing spaces, if any
    return table



def checkIntersection(listA, listB):
    for i, j in enumerate(listA):
        listA[i] = tuple(sorted(j))
    for i, j in enumerate(listB):
        listB[i] = tuple(sorted(j))
    listA = set(listA)
    listB = set(listB)
    return len(list(listA.intersection(listB)))

    
def PDELAYFD(table_list, atable):
    #print(table_list)
    #table_list = glob.glob(r"minimum_example/*.csv")
    #table_list = []
    #temp = "a_table1.csv"
    #table_list.append(temp)
    all_output_tuples = []
    arbitrary_table_path = atable
    print("arbitrary table:", arbitrary_table_path)
    table = pd.read_csv(arbitrary_table_path, encoding='latin1',
                        error_bad_lines="false")
    
    table =  preprocess(table)
    #row_attributes = set(list(table.columns))
    q_rx = []
    q_rx_hashed = set()
    dict_of_all_rows = {}
    all_rows = table.to_dict(orient='records')
    dict_of_all_rows[arbitrary_table_path] = all_rows
    all_selected_rows = []
    for file in table_list: #line 8 start
        if file != arbitrary_table_path:
            selected_table = pd.read_csv(file, encoding='latin1', error_bad_lines = "false")
            selected_table =  preprocess(selected_table)
            all_selected_rows += selected_table.to_dict(orient='records')
            dict_of_all_rows[file] = selected_table.to_dict(orient='records')
    #print(all_rows)
    print("--------------------")
    all_jcc_time = []
    all_extend_time = []
    previous_extended_sets = {}
    all_columns = set()
    for file in table_list:    
        df= pd.read_csv(file, nrows=0, encoding = "latin1")
        current_columns = set(df.columns)
        for col in current_columns:
            all_columns.add(col.lower())
    all_columns = list(all_columns) #outcome of line 2
    #table_rows = table.to_records(index = False).tolist()
    start_cut_off = time.time()
    current_r_num = 0
    total_selected_rows = len(all_selected_rows)
    total_r_rows = len(all_rows)
    for t in all_rows:
        current_r_num += 1
        print("r =", current_r_num, "/",total_r_rows,"  (total s):", total_selected_rows)
        #print("t = ", t)
        q, q_hashed, current_output_tuples, jcc_time, ext_time, new_extended_set = TUPEXTFD(table_list, arbitrary_table_path, t, q_rx, q_rx_hashed, all_selected_rows, dict_of_all_rows, previous_extended_sets, all_columns)
        #q, current_output_tuples, current_c_tx, current_q_t = TUPEXTFD(table_list, arbitrary_table_path, t, all_q_t, q_rx)
        #print("q:",q)
        #print("current output tuples:",current_output_tuples)
        #count = 0
        if jcc_time == "late":
            return "late"
        if int(time.time() - start_cut_off) > 10100:
            return "late"
        previous_extended_sets = new_extended_set
        q_rx = q
        q_rx_hashed = q_hashed
        all_jcc_time += jcc_time
        all_extend_time += ext_time
        if len(current_output_tuples) >0:
            for item in current_output_tuples:
                all_output_tuples.append(item)
    #print(all_output_tuples)
    #print("q_r:",q_rx)
    #print("c_tx: ",c_tx)
    print("current output size:", len(all_output_tuples))
    print("Relex size:", len(q_rx))
    #print()
    # for item in q_rx:
    #     print(item)
    q, current_output_tuples = RELEXCFD(table_list, arbitrary_table_path, q_rx, q_rx_hashed, all_selected_rows, dict_of_all_rows, previous_extended_sets, all_columns)
    if q == "late":
        return "late"
    if len(q) ==0:
        print("all processed")
        #print("current op tuples", current_output_tuples)
    else:
        print("bug detected in the code.")
    df_tuplex = pd.DataFrame(all_output_tuples)
    df_relex = pd.DataFrame(current_output_tuples)
    df_merged = pd.concat([df_tuplex, df_relex])
    df_merged.reset_index(drop=True, inplace=True)
    #return {"tuplex":all_output_tuples, "relex":current_output_tuples}, all_jcc_time, all_extend_time
    return df_merged
    if len(current_output_tuples) > 0:
        for item in current_output_tuples:
            #print(item)
            all_output_tuples.append(item)
    else:
        print("relex returned nothing")
    final_df_result = pd.DataFrame(all_output_tuples)
    final_df_result = final_df_result.drop_duplicates()
    return final_df_result, all_jcc_time, all_extend_time

def embeds(tuple_dict, schema_list):
    output_tuple = {}
    for current_tuple in tuple_dict:
        for key in current_tuple:
            output_tuple[key] = current_tuple[key]
    for each in schema_list:
        if each not in output_tuple:
            output_tuple[each] = "nan"
    return output_tuple


def CheckIfExistsOld(setOfTuples, listOfListOfDict, schema):
    embedCheckTuple = tuple(sorted(tuple(embeds(setOfTuples, schema).items()), key = itemgetter(0)))
    embedAllTuple = set()
    #print(len(listOfListOfDict))
    for each in listOfListOfDict:
        temp_result = tuple(sorted(tuple(embeds(each, schema).items()), key = itemgetter(0)))
        #print(temp_result)
        embedAllTuple.add(temp_result)
    if embedCheckTuple in embedAllTuple:
        return 1
    else:
        return 0
    
def CheckIfExists(setOfTuples, embedAllTuple, schema):
    embedCheckTuple = tuple(sorted(tuple(embeds(setOfTuples, schema).items()), key = itemgetter(0)))
    if embedCheckTuple in embedAllTuple:
        return 1
    else:
        return 0
    


def HashTupleList(tl):
    hashed = []
    for every in tl:
        hashed.append(tuple(sorted(tuple(every.items()), key = itemgetter(0))))
    return tuple(sorted(hashed))

#new = [{'state': 'kansas', 'rank': '50', 'term start': 'january 13, 2020'},
 #{'state': 'kansas', 'population': '3344'}]
#see = HashTupleList(new)
        
def TUPEXTFD(table_list, table_path, t, receive_q_rx, receive_q_rx_hashed, all_selected_rows, dict_of_all_rows, receive_extended_sets, all_columns):
    #print("Receive q rx", receive_q_rx)
    start_cut_off_time = time.time()
    output_list = []
    q_r_next = receive_q_rx
    q_r_next_hashed = receive_q_rx_hashed #set

    current_table_tuple_set = set()
    tuples_in_table = dict_of_all_rows[table_path]
    for every_t_dict in tuples_in_table:
        current_table_tuple_set.add(tuple(sorted(tuple(every_t_dict.items()), key = itemgetter(0))))
        #current_table_tuple_set.add(tuple(every_t_dict.items()))
    q_t = []
    c_t = [] #line 1
    q_t_hashed = set()
    c_t_hashed = set()
    q_t_c_t_hashed = set()
    max_jcc_time = []
    extend_time = []
    tuple_list = []
    tuple_list.append(t)
    tuple_table_set = set()
    tuple_table_set.add(table_path)
    extended_tuple_dict, tuple_table_set = EXTENDTOMAX(table_list, tuple_list, tuple_table_set, dict_of_all_rows)
    q_t.append(extended_tuple_dict) #line 2
    extended_tuple_dict_hashed = tuple(sorted(tuple(embeds(extended_tuple_dict, all_columns).items()), key = itemgetter(0)))
    q_t_hashed.add(extended_tuple_dict_hashed)
    q_t_c_t_hashed.add(extended_tuple_dict_hashed)
    while len(q_t) > 0: #line 4
        #profile = cProfile.Profile()
        #profile.enable()
        ready_to_print = q_t.pop(0) #line 5
        T = ready_to_print
        #print("rtp", ready_to_print)
        c_t.append(ready_to_print) #line 7
        ready_to_print_hashed = tuple(sorted(tuple(embeds(ready_to_print, all_columns).items()), key = itemgetter(0)))
        c_t_hashed.add(ready_to_print_hashed)
        #q_t_c_t_hashed.add(ready_to_print_hashed)
        ready_to_print = embeds(ready_to_print, all_columns)
        
        output_list.append(ready_to_print) #line 6
        ready_to_print = tuple(sorted(tuple(ready_to_print.items()), key = itemgetter(0)))
        
# =============================================================================
        #print("All selected rows (s):", len(all_selected_rows))
        #cc = 0
        #print("len q_T", len(q_t))
        for s in all_selected_rows: #line 8 loop
            if int(time.time() - start_cut_off_time) > 5000:
                return q_r_next, q_r_next_hashed, output_list, "late", extend_time, receive_extended_sets
            #cc += 1
            #print("s = ", cc)
            temp_ts = T
            #temp_ts = MakeJCC(temp_ts, s, selected_rows_in_tuple_form)
            #start_time = time.time_ns()
            #print("temp ts before:", temp_ts)
            temp_ts = MakeJCC(temp_ts, s)
            #test_print = tuple(sorted(tuple(temp_ts.items()), key = itemgetter(0)))
            #print("temp ts after jcc:", temp_ts)
            #print("temp ts after jcc hashed:", test_print)
            hashed_temp_ts = HashTupleList(temp_ts)
            if hashed_temp_ts in receive_extended_sets:
                hashed_list = receive_extended_sets[hashed_temp_ts]
                extended_tuple_s = hashed_list[0]
                tuple_table_set_s = hashed_list[1]
            else:
                extended_tuple_s, tuple_table_set_s = EXTENDTOMAX(table_list, temp_ts, tuple_table_set, dict_of_all_rows)
                receive_extended_sets[hashed_temp_ts] = [extended_tuple_s, tuple_table_set_s]
            extended_tuple_set = set()
            for ex in extended_tuple_s:
                extended_tuple_set.add(tuple(sorted(tuple(ex.items()), key = itemgetter(0))))
                #extended_tuple_set.add(tuple(ex.items()))
            #print("extention stored:",extended_tuple_set)
            before_extension_set = set()
            for y in T:
                before_extension_set.add(tuple(sorted(tuple(y.items()), key = itemgetter(0))))
            extended_tuple_s_hashed = tuple(sorted(tuple(embeds(extended_tuple_s, all_columns).items()), key = itemgetter(0)))
            
            if tuple(sorted(tuple(t.items()), key = itemgetter(0))) in extended_tuple_set and extended_tuple_s_hashed not in q_t_c_t_hashed:
            #and CheckIfExists(extended_tuple_s, c_t_hashed.union(q_t_hashed), all_columns) == 0:
                q_t.append(extended_tuple_s)
                q_t_hashed.add(extended_tuple_s_hashed)
                q_t_c_t_hashed.add(extended_tuple_s_hashed)
            if len(extended_tuple_set.intersection(current_table_tuple_set)) == 0 and CheckIfExists(extended_tuple_s, q_r_next_hashed, all_columns) == 0:
                q_r_next.append(extended_tuple_s)
                q_r_next_hashed.add(extended_tuple_s_hashed)
                
                
    return q_r_next, q_r_next_hashed, output_list, max_jcc_time, extend_time, receive_extended_sets

def JCC(new_t, existing_t):
    #check what are the matching positions:
    existing_t_schema = set()
    for t in existing_t:
        current_t_schema = set(t.keys())
        existing_t_schema = existing_t_schema.union(current_t_schema)
    new_t_schema = set(new_t.keys())
    intersecting_attributes = new_t_schema.intersection(existing_t_schema)
    if len(intersecting_attributes) > 0:
        for t in existing_t:
            current_t_schema = set(t.keys())
            for a in intersecting_attributes:
                if a in current_t_schema and (t[a] != new_t[a] or (t[a] == new_t[a] and t[a] == "nan")):
                        return 0
        return 1
    else:
        return 0

def MakeJCC(tuple_set, tuple_s):
    #tuple_set.append(tuple_s)
    tuple_s_schema = set(tuple_s.keys())
    final_schema = tuple_s_schema
    updated_t_set = []
    possible_t_set = []
    for tuple_dict in tuple_set:
        current_dict_attributes = set(tuple_dict.keys())
        intersecting_attributes = current_dict_attributes.intersection(tuple_s_schema)
        if len(intersecting_attributes) > 0:
            status = 1
            for a in intersecting_attributes:
                if tuple_s[a] != tuple_dict[a] or (tuple_s[a] == tuple_dict[a] and tuple_s[a] == "nan"):
                    status = 0
                    break
            if status == 1:
                updated_t_set.append(tuple_dict)
                final_schema = final_schema.union(current_dict_attributes)
        else:
            possible_t_set.append(tuple_dict)
    
    for tuple_dict in possible_t_set:
        if len(final_schema.intersection(set(tuple_dict.keys()))) > 0:
            updated_t_set.append(tuple_dict)
    updated_t_set.append(tuple_s)
    return updated_t_set




def RELEXCFD(table_list, table_path, q_r, q_r_hashed, all_selected_rows, dict_of_all_rows, receive_extended_sets, all_columns):
    print("Relex start")
    start_cut_off_time = time.time()
    output_list = []
    table = pd.read_csv(table_path, encoding='latin1', error_bad_lines="false")
    table =  preprocess(table)
    #table_attributes = set(list(table.columns))
    #tuples_in_table = table.values.tolist()
    current_table_tuple_set = set()
    tuples_in_table = table.to_dict(orient='records')
    for every_t_dict in tuples_in_table:
        current_table_tuple_set.add(tuple(sorted(tuple(every_t_dict.items()), key = itemgetter(0))))
    #print("current table tuple set:")
    #t = {'stadium': 'nrg stadium', 'location': 'texas', 'team': 'houston texans'}
    #table_list = ['minimum_example\\t1.csv', 'minimum_example\\t2.csv', 'minimum_example\\t3.csv', 'minimum_example\\t4.csv', 'minimum_example\\t5.csv', 'a_table1.csv']
    #table_path = r"minimum_example\t1.csv"
    c_r = [] #line 1
    c_r_hashed = set()
    c_r_q_r_hash_union = q_r_hashed
    #print(q_t)
    count_prints = 0
    #print("All columns:",all_columns)
    #print(type(q_t))
    # for item in q_r_hashed:
    #     print(item)
    count_relex_size = 0
    while len(q_r) > 0: #line 4
        if int(time.time() - start_cut_off_time) > 10100:
            return "late", output_list
        ready_to_print = q_r.pop(0) #line 5
        T = ready_to_print
        c_r.append(ready_to_print) #line 7
        ready_to_print_hashed = tuple(sorted(tuple(embeds(ready_to_print, all_columns).items()), key = itemgetter(0)))
        c_r_hashed.add(ready_to_print_hashed)
        c_r_q_r_hash_union.add(ready_to_print_hashed)
        ready_to_print = embeds(ready_to_print, all_columns)
        #print(ready_to_print)
        output_list.append(ready_to_print) #line 6
        ready_to_print = tuple(sorted(tuple(ready_to_print.items()), key = itemgetter(0)))
        #print(ready_to_print)
        count_prints += 1
        if (count_prints) % 100 == 0:
            print("printed relex tuples:",(count_prints))
        #selected_schema = set(selected_table.columns)
        included_table_set = set()
        for s in all_selected_rows: #line 8 loop
            temp_ts = T
            #print("temp ts before jcc", tuple(sorted(tuple(temp_ts.items()), key = itemgetter(0))))
            temp_ts = MakeJCC(temp_ts, s)
            #print("temp ts after jcc", temp_ts)
            included_table_set.add(file)
            hashed_temp_ts = HashTupleList(temp_ts)
            #print("before extension:", hashed_temp_ts)
            if hashed_temp_ts in receive_extended_sets:
                hashed_list = receive_extended_sets[hashed_temp_ts]
                extended_tuple_s = hashed_list[0]
                tuple_table_set_s = hashed_list[1]
                #print('used from hash')
            else:
                extended_tuple_s, tuple_table_set_s = EXTENDTOMAX(table_list, temp_ts, included_table_set, dict_of_all_rows)
                receive_extended_sets[hashed_temp_ts] = [extended_tuple_s, tuple_table_set_s]
                #print("used from scratch")
                #print("tuple:", hashed_temp_ts)
            #print("temp ts after extension",  tuple(sorted(tuple(extended_tuple_s.items()), key = itemgetter(0))))
            #print("--------------------------------")
            extended_tuple_set = set()
            for ex in extended_tuple_s:
                extended_tuple_set.add(tuple(sorted(tuple(ex.items()), key = itemgetter(0))))
            
            extended_tuple_s_hashed = tuple(sorted(tuple(embeds(extended_tuple_s, all_columns).items()), key = itemgetter(0)))
            #print("extended tuple set:", extended_tuple_set)
            if len(extended_tuple_set.intersection(current_table_tuple_set)) == 0 and extended_tuple_s_hashed not in c_r_q_r_hash_union:
            #CheckIfExists(extended_tuple_s, q_r_hashed.union(c_r_hashed), all_columns) == 0:
                #print("here")
                count_relex_size += 1
                q_r.append(extended_tuple_s)
                q_r_hashed.add(extended_tuple_s_hashed)
    print("relex extended by:", count_relex_size)
    #print("current table tuple set:", current_table_tuple_set)
    return q_r, output_list


def EXTENDTOMAX(table_list, tuple_list, tuple_table_set, dict_of_all_rows):
    #table_list = ['minimum_example\\t1.csv', 'minimum_example\\t2.csv', 'minimum_example\\t3.csv', 'minimum_example\\t4.csv', 'minimum_example\\t5.csv']
    #tuple_list = [{'stadium': 'nrg stadium', 'location': 'texas', 'team': 'houston texans'}]
    #print(tuple_table_set)
    visited = {}
    tuple_schema = set()
    tuple_set = set()
    for t in tuple_list:
        tuple_set.add(tuple(sorted(tuple(t.items()), key = itemgetter(0))))
        tuple_schema = tuple_schema.union(set(t.keys()))
    
    #tuple_list_new will only be updated
    tuple_input_dict = tuple_list
    all_table_schema = {}
    #line 2 to 4 start
    for table_path in table_list:
        current_table_tuple_set = set()
        tuples_in_table = dict_of_all_rows[table_path]
        all_table_schema[table_path] = set(tuples_in_table[0].keys())
        for every_t_dict in tuples_in_table:
            current_table_tuple_set.add(tuple(sorted(tuple(every_t_dict.items()), key = itemgetter(0))))
        #current_tuple_schema = set(table.columns)
        #print("current table tuple set:", current_table_tuple_set)
        #print("tuple set", tuple_set)
        if len(current_table_tuple_set.intersection(tuple_set)) == 0: # and table_path not in tuple_table_set:
            visited[table_path] = 0
        else:
            visited[table_path] = 1
        #while part
    #line 2 to 4 end
    while (1):
        #print(visited)
        #print("below visited:",tuple_input_dict)
        false_count = 0
        for each_table in visited:
            if visited[each_table] == 0:
                current_selected_rows = dict_of_all_rows[each_table]
                selected_schema = all_table_schema[each_table]
                if len(tuple_schema.intersection(selected_schema)) > 0:
                    false_count += 1
                    visited[each_table] = 1 #line 6
                    break
        #print("each table:", each_table)
        #print("here")
        #print("false count:", false_count)
        if false_count == 0: #line 5 first condition check
            #print("here")
            #print(each_table)
            return tuple_input_dict, tuple_table_set #while loop termination if 
                                    #no such table exists
        
        
        for t in current_selected_rows:
            flag = JCC(t,tuple_input_dict)
            if flag == 1:
                tuple_input_dict.append(t)
                tuple_schema = tuple_schema.union(set(t.keys()))
                tuple_table_set.add(each_table)
                break
        
    print("Extend to max", tuple_input_dict)
    return tuple_input_dict, tuple_table_set
                    


#BICOMNLOJ is applied  in the main function
if __name__ == "__main__":
    #INPUT_TABLE_PATH = r"."
    time_stats = dict()
    output_results = {}
    print("Enter input folder path:")    
    #input_path = str(input())
    input_path = r"minimum_example/"
    print(input_path)
    output_path = r"output_tables/poly_delay/"+ input_path
# =============================================================================
#     if not os.path.exists(output_path):
#       # Create a new directory because it does not exist 
#       os.makedirs(output_path)
#       print("output directory is is created!")
# =============================================================================
    stat_folder = r"statistics/poly_delay/"
# =============================================================================
#     if not os.path.exists(stat_folder):
#       # Create a new directory because it does not exist 
#       os.makedirs(stat_folder)
#       print("stat directory is is created!")
# =============================================================================
    
    stat_path = stat_folder+ input_path[:-1]+".csv"
    foldernames = glob.glob(input_path + "*")
    statistics = pd.DataFrame(
            columns = ["cluster", "n", "f",
                       "produced_nulls", "biconnected_components",
                       "subsume_time",
                       "subsumed_tuples", "total_time"])
    record_bcc_numbers = {}
    for cluster in foldernames:
        try:
            filenames = glob.glob(cluster + "/*.csv")
        except:
            continue
        cluster_name = cluster.rsplit(os.sep)[-1]
        m = len(filenames)
        all_columns_order = set()
# =============================================================================
#             create scheme graphs
# =============================================================================
        start_time = time.time_ns()        
        order = [] #schema of each table
        for file in filenames:    
            df= pd.read_csv(file, nrows=0, encoding = "latin1")
            df.columns = map(str.lower, df.columns)
            order.append(set(df.columns))
            all_columns_order = all_columns_order.union(set(df.columns))
        total_tables = len(filenames)
        all_columns_order = list(all_columns_order)
        biconnected_components = []
        connections = set()
        #np.fill_diagonal(scheme_graph, 1)
        for i in range(0, total_tables-1):
            for j in range(i, total_tables):
                if i != j and len(order[i].intersection(order[j]))>0:
                    connections.add(tuple(sorted((i, j))))
        bc = bcc.Graph(total_tables)
        for connect in connections:
            bc.addEdge(connect[0], connect[1])
        bc.AP()
        print("-----------------------------\n")
        print("cluster:", cluster_name)
        print("articulation points:",bc.articulation_points)
        bc.BCC();
        print ("Above are % d biconnected components in graph" %(bc.count));
        biconnected_components = bc.biconnected_components
        articulation_points = set(bc.articulation_points)
        track_articulation_files = set()
        for point in articulation_points:
            track_articulation_files.add(filenames[point])
            
        bcc_table_ids = []
        tables_in_bcc = set()
        for biconnected_component in biconnected_components:
            current_tables = set()
            for edge in biconnected_component:
                if len(biconnected_component) > 1:
                    current_tables.add(edge[0])
                    current_tables.add(edge[1])
                else:
                    if edge[0] not in articulation_points:
                        current_tables.add(edge[0])
                    if edge[1] not in articulation_points:
                        current_tables.add(edge[1])
            if len(current_tables) > 0:
                bcc_table_ids.append(current_tables)
                tables_in_bcc = tables_in_bcc.union(current_tables)
        for i in range(0, total_tables):
            if i not in tables_in_bcc:
                bcc_table_ids.append({i})
        bcc_schemas = [] #schema of each bcc
        for each in bcc_table_ids:
            current_scheme = set()
            for tid in each:
                current_scheme = current_scheme.union(order[tid])
            bcc_schemas.append(current_scheme)
        record_bcc_numbers[cluster] = len(bcc_schemas)
        
        #find strongly connected components
        bcc_connections  = set()
        
        #make scc graph
        total_bc_components = len(bcc_table_ids) 
        for i in range(0, total_bc_components-1):
            for j in range(i, total_bc_components):
                if i != j and len(bcc_schemas[i].intersection(bcc_schemas[j]))>0:
                    bcc_connections.add(tuple(sorted((i, j))))
        sc = scc.Graph(total_bc_components)
        for connect in bcc_connections:
            sc.addEdge(connect[0], connect[1])
        scc_ordering = sc.printSCCs()
        print("-----------------------------\n")
# =============================================================================
#         if cluster_name == "R6":
#             break
#         else:
#             continue
# =============================================================================
        #start full disjunction
        null_set = set()
        null_count = 0 
        ordered_filenames= []
        for bcc_id in scc_ordering:
            get_bcc_table_ids = bcc_table_ids[bcc_id]
            prepare_table_list = []
            for bcc_tid in get_bcc_table_ids:
                prepare_table_list.append(filenames[bcc_tid])
            ordered_filenames.append(prepare_table_list)
        
        #prepare first table for applying left deep outer join
        first_bcc = ordered_filenames.pop(0)
        #if more than  1 table, apply poly delay, else apply outer join
        if len(first_bcc) > 1:
            connecting_table = list(track_articulation_files.intersection(set(first_bcc)))
            if len(connecting_table) > 0:
                #jc time and ext time not needed anywhere, just for debugging
                #final_outcome, jc_time, ext_time = PDELAYFD(file_list, connecting_table[0])
                first_table = PDELAYFD(first_bcc, connecting_table[0])
            else:
                first_table = PDELAYFD(first_bcc, first_bcc[0])
            if isinstance(first_table, str) == True:
                append_list = [cluster.rsplit(os.sep)[-1], total_tables, "nan", "nan", len(bcc_table_ids), "nan", "nan", "first cut off"]
                a_series = pd.Series(append_list, index = statistics.columns)
                statistics = statistics.append(a_series, ignore_index=True)
                statistics.to_csv(stat_path, index = False)
                continue
        else:
            first_table = pd.read_csv(first_bcc[0], encoding='latin1', error_bad_lines="false")
            first_table =  preprocess(first_table) 
        first_table = first_table.replace("nan",np.nan)
        if first_table.isnull().sum().sum() > 0:
            #print(filenames[0])
            first_table, null_count, current_null_set = cfd.ReplaceNulls(first_table, null_count)
            null_set = null_set.union(current_null_set)
        break_stat = 0
        #apply polydelay FD algorithm for each bcc and outer join the results
        for file_list in ordered_filenames:
            if len(file_list) > 1:
                connecting_table = list(track_articulation_files.intersection(set(file_list)))
                if len(connecting_table) > 0:
                    second_table = PDELAYFD(file_list, connecting_table[0])
                else:
                    second_table = PDELAYFD(file_list, file_list[0])
                if isinstance(second_table, str) == True:
                    append_list = [cluster.rsplit(os.sep)[-1], total_tables, "nan", "nan", len(bcc_table_ids), "nan", "nan", "second cut off"]
                    a_series = pd.Series(append_list, index = statistics.columns)
                    statistics = statistics.append(a_series, ignore_index=True)
                    statistics.to_csv(stat_path, index = False)
                    break_stat = 1
                    break
            else:
                second_table = pd.read_csv(file_list[0], encoding='latin1', error_bad_lines="false")
                second_table =  preprocess(second_table)
            second_table = second_table.replace("nan",np.nan)
            if second_table.isnull().sum().sum() > 0:
                #print(filenames[0])
                second_table, null_count, current_null_set = cfd.ReplaceNulls(second_table, null_count)
                null_set = null_set.union(current_null_set)
            first_table = first_table.merge(second_table, how = "outer", on = None)
            if first_table.isnull().sum().sum() > 0:
                        #print(filenames[0])
                        first_table, null_count, current_null_set = cfd.ReplaceNulls(first_table, null_count)
                        null_set = null_set.union(current_null_set)
        if break_stat == 1:
            break_stat = 0
            continue
        print("Adding nulls back...")
        if len(null_set) > 0:
                first_table =  cfd.AddNullsBack(first_table, null_set)
        print("Added nulls back...")
        fd_data = {tuple(x) for x in first_table.values}
        start_subsume_time = time.time_ns()
        print("Output tuples before subsumption: ( total", len(fd_data),")")
        for t in fd_data:
            print(t)
        print("----------------------------")
        subsumptionResults = cfd.EfficientSubsumption(list(fd_data))
        print("Output tuples after subsumption: ( total", len(subsumptionResults),")")
        for t in subsumptionResults:
            print(t)
        end_time = time.time_ns()
        subsume_time = int(end_time - start_subsume_time)/ 10**9
        total_time = int(end_time - start_time)/ 10**9
        subsumed_tuples = len(list(fd_data)) - len(subsumptionResults)
        append_list = [cluster.rsplit(os.sep)[-1], total_tables, len(subsumptionResults),
                       len(null_set), len(bcc_table_ids), subsume_time, subsumed_tuples, total_time]
        a_series = pd.Series(append_list, index = statistics.columns)
        statistics = statistics.append(a_series, ignore_index=True)
        #statistics.to_csv(stat_path, index = False)
        