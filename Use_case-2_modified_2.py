Shortcuts simplify My Drive … 
In the coming weeks, items in more than one folder will be replaced by shortcuts. Access to files and folders won't change.Learn more
#Use case-2 modified
import os, sys
from pathlib import Path
sys.path.append(os.path.dirname(sys.path[0]))
from hw2vec.config import Config
from hw2vec.hw2graph import *
from hw2vec.graph2vec.models import *
#--yaml_path ./example_gnn4tj.yaml --raw_dataset_path ../assets/TJ-RTL-toy --data_pkl_path dfg_tj_rtl.pkl --graph_type DFG (--device cuda)
cfg1 = Config(sys.argv[1:])

# cfg2 = Config(str("--yaml_path ./example_gnn4tj.yaml --raw_dataset_path ../assets/TJ-RTL-toy --data_pkl_path merged_tj_rtl_DFG_AST.pkl --graph_type AST"))
''' prepare graph data '''
nx_graphs = []
if not cfg1.data_pkl_path.exists():
    ''' converting graph using hw2graph '''
    hw2graph = HW2GRAPH(cfg1)
    for hw_project_path in hw2graph.find_hw_project_folders():
        hw_graph = hw2graph.code2graph(hw_project_path)
        nx_graphs.append(hw_graph)
    
    data_proc1 = DataProcessor(cfg1)
    for hw_graph in nx_graphs:
        data_proc1.process(hw_graph)
    data_proc1.cache_graph_data(cfg1.data_pkl_path)
    
else:
    ''' reading graph data from cache '''
    data_proc1 = DataProcessor(cfg1)
    data_proc1.read_graph_data_from_cache(cfg1.data_pkl_path)

cfg2=cfg1
cfg2.data_pkl_path=Path("ast_tj_rtl_AST.pkl").resolve()
cfg2.graph_type="AST"  

nx_graphs2=[]  
if not cfg2.data_pkl_path.exists():
    ''' converting graph using hw2graph '''
    hw2graph = HW2GRAPH(cfg2)
    for hw_project_path in hw2graph.find_hw_project_folders():
        hw_graph = hw2graph.code2graph(hw_project_path)
        nx_graphs2.append(hw_graph)
    
    data_proc2 = DataProcessor(cfg2)
    for hw_graph in nx_graphs2:
        data_proc2.process(hw_graph)
    data_proc2.cache_graph_data(cfg2.data_pkl_path)
    
else:
    ''' reading graph data from cache '''
    data_proc2 = DataProcessor(cfg2)
    data_proc2.read_graph_data_from_cache(cfg2.data_pkl_path)
# print("Merged Data")
# for i in nx_graphs2:
#   nx_graphs.append(i)
# print(len(nx_graphs))
''' prepare dataset '''
TROJAN = 1
NON_TROJAN = 0

all_graphs_DFG = data_proc1.get_graphs()
print(all_graphs_DFG)
for data in all_graphs_DFG:
    if "TjFree" == data.hw_type:
        data.label = NON_TROJAN
    else:
        data.label = TROJAN

all_graphs_AST = data_proc2.get_graphs()
print(all_graphs_AST)
for data in all_graphs_AST:
    if "TjFree" == data.hw_type:
        data.label = NON_TROJAN
    else:
        data.label = TROJAN

train_graphs_DFG, test_graphs_DFG = data_proc1.split_dataset(ratio=cfg1.ratio, seed=cfg1.seed, dataset=all_graphs_DFG)
train_graphs_AST, test_graphs_AST = data_proc2.split_dataset(ratio=cfg2.ratio, seed=cfg2.seed, dataset=all_graphs_AST)
train_loader_DFG = DataLoader(train_graphs_DFG, shuffle=True, batch_size=cfg1.batch_size)
valid_loader_DFG = DataLoader(test_graphs_DFG, shuffle=True, batch_size=1)
train_loader_AST = DataLoader(train_graphs_AST, shuffle=True, batch_size=cfg2.batch_size)
valid_loader_AST = DataLoader(test_graphs_AST, shuffle=True, batch_size=1)

''' model configuration '''
model = GRAPH2VEC(cfg2)
if cfg2.model_path != "":
    model_path = Path(cfg2.model_path)
    if model_path.exists():
        model.load_model(str(model_path/"model.cfg"), str(model_path/"model.pth"))
else:
    convs = [
        GRAPH_CONV("gcn", data_proc2.num_node_labels, cfg2.hidden),
        GRAPH_CONV("gcn", cfg2.hidden, cfg2.hidden)
    ]
    model.set_graph_conv(convs)

    pool = GRAPH_POOL("sagpool", cfg2.hidden, cfg2.poolratio)
    model.set_graph_pool(pool)

    readout = GRAPH_READOUT("max")
    model.set_graph_readout(readout)

    output = nn.Linear(cfg2.hidden, cfg2.embed_dim)
    model.set_output_layer(output)

''' training '''
model.to(cfg2.device)
trainer = GraphTrainer(cfg2, class_weights=data_proc2.get_class_weights(train_graphs_AST))
trainer.build(model)
trainer.train(train_loader_AST, valid_loader_AST)
trainer.train(train_loader_DFG, valid_loader_DFG)
''' evaluating and inspecting '''
trainer.evaluate(cfg1.epochs, train_loader_DFG, valid_loader_DFG)
trainer.evaluate(cfg2.epochs, train_loader_AST, valid_loader_AST)
#vis_loader_1 = DataLoader(all_graphs_DFG, shuffle=False, batch_size=1)
#vis_loader_2 = Dataloader(all_graphs_AST, shuffle=False, batch_size=1)
#trainer.visualize_embeddings(vis_loader_1, "./")