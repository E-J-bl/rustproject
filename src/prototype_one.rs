use std::cmp::Ordering;
use std::collections::HashMap;
use std::{fmt};
use std::fs::File;
use std::io::Write;
use std::fs::OpenOptions;
//performance note i will be using Arc<Mutex<T>> over Rc<RefCell<T>>
// despite a performace cost that this will occur it will reduce the code overhaul
// i would have to do if i later decide to thread the process
use std::sync::{Arc, Mutex};

struct Graph {
    nodes: Vec<i32>,
    edges: Vec<Vec<i32>>,
    live_map: Vec<i32>,
    node_copy: Vec<i32>,
    been_checked: bool,
    is_connected: bool,
}

impl Graph {
    fn new(all_nodes: Vec<i32>,all_edges: HashMap<i32,Vec<i32>>) -> Self {
        if all_nodes.len() !=all_edges.iter().len(){
            panic!("there must be the same number of nodes as node edges");
        }

        let nodes = all_nodes.clone();
        let nodescop= all_nodes.clone();
        let most_connections=all_edges
            .values()
            .max_by_key(|v| v.len())
            .unwrap().len();

        println!("{}",most_connections);
        let mut count=0;
        let mut edges=vec![vec![0;most_connections as usize];nodes.len()];
        for key in all_edges.keys(){
            for value in all_edges[key].iter(){
                edges[*key as usize -1][count]=*value;
                count+=1;
            }
            count=0
        }
        Self {nodes,edges,live_map:vec![],node_copy: nodescop,been_checked:false,is_connected:false}

    }

    fn solve(&mut self)-> bool {
        if !self.been_checked {
            let start=match self.edges[self.nodes[0] as usize ].iter().sum::<i32>().cmp(&0){
                Ordering::Equal=>panic!("there is a floating node"),
                Ordering::Greater=>self.nodes[0],
                Ordering::Less=>self.nodes[0]
            };
            self.nodes[start as usize-1]=0;
            self.live_map.push(start);
            //println!("{}",self);
            while self.live_map.len() > 0{
                self.step();
                //println!("{}",self);
                //io::stdin().read(&mut [0u8]).unwrap();
            }
            if self.nodes.iter().sum::<i32>()>0{
                self.been_checked = true;
                return false;
            }else{
                self.been_checked = true;
                self.is_connected = true;
                return true;
            }

        }
        return self.is_connected;
    }

    fn step(&mut self){

        let active= self.live_map.pop().unwrap();

        for con in self.edges[active as usize-1].iter(){
            if *con==0{break}
            if self.nodes[*con as usize-1] != 0 {
                if self.nodes[*con as usize-1]<0{
                    self.live_map.push(*con);
                }
                self.nodes[*con as usize -1] =0;
            }

        }


    }

    fn from_json(file_path: &str) ->Self {
        let file = File::open(file_path).expect("file not found");
        let json: serde_json::Value = serde_json::from_reader(file).expect("file should be proper JSON");
        let mut noderes=Vec::<i32>::new();
        let nodes = json["Node_list"].as_array().unwrap();
        let edges = json["Edge_list"].as_array().unwrap();
        let edgeres: Vec<[i32; 2]>= edges
            .iter()
            .map(|edge| {
                edge.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i32)
                    .collect::<Vec<i32>>()
                    .try_into()
                    .expect("Each edge must have exactly 2 elements")
            })
            .collect();

        let mut edg_final:HashMap<i32,Vec<i32>>=HashMap::new();

        //println!("{:?} {:?}",edges,nodes);
        let mut is_router:bool;
        for node in nodes {
            is_router=json["Which_are_router"][node.as_str().unwrap()].as_bool().unwrap();

            noderes.push(node.as_str().unwrap().parse::<i32>().unwrap()*(1+(-2)*is_router as i32));
        }
        for edge in edgeres {
            edg_final.entry(edge[0]).or_insert_with(Vec::new).push(edge[1]);
            edg_final.entry(edge[1]).or_insert_with(Vec::new).push(edge[0]);
        }
        //println!("{:?}",noderes);
        //println!("{:?}",edgeres);
        Graph::new( noderes, edg_final,)
    }

    fn find_routers (&mut self)-> Vec<i32>{
        vec![0]
    }
    fn solution_output(&self, filepath: &str){
        let mut file = OpenOptions::new()
            .write(true)
            .append(true) // or .truncate(true) if you want to overwrite
            .open(filepath)
            .expect("could not open file for writing");

        for node in &self.node_copy{
            //println!("{}",node);
            if *node<0{
                file.write(
                    format!("{}\n",node.abs().to_string()).as_bytes())
                    .expect("could not write");
            }
        }

    }

}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let edg=self.edges.iter().map(
            |x|x.iter().map(
                |s| s.to_string()).collect::<Vec<String>>().join(" "))
            .collect::<Vec<String>>().join("\n");
        write!(f,"{:?}\n {} \n {:?}",self.nodes,edg,self.live_map)
    }
}
fn main() {
    let mut gh: Graph= Graph::new(vec![1,2,3,4],HashMap::from([
        (1,vec![4,2]),
        (2,vec![1,3]),
        (3,vec![4,2]),
        (4,vec![1,3])
    ]));
    println!("{}",gh);
    println!("{}",gh.solve());
    let mut test_gh: Graph= Graph::from_json("src/for_testing.json");
    println!("{}",test_gh);
    println!("{}",test_gh.solve());
    test_gh.solution_output("src/output.txt");
}
