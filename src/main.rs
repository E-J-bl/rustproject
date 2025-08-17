use std::collections::HashMap;
use std::{fmt};
use std::fs::File;
//performance note i will be using Arc<Mutex<T>> over Rc<RefCell<T>>
// despite a performance cost that this will occur it will reduce the code overhaul
// i would have to do if i later decide to thread the process
use std::sync::{Arc, Mutex};


struct Graph{
    work_space:Vec<Arc<Mutex<i64>>>,
    record_space:Vec<Arc<Mutex<i64>>>,
    map_space:Vec<Vec<Arc<Mutex<i64>>>>,
    live_stack:Vec<Arc<Mutex<i64>>>,
}
fn true_len(node: &Vec<Arc<Mutex<i64>>>) -> u64 {
    let mut ln=0;
    for i in node{
        if *i.lock().unwrap()!=0{
            ln+=1
        }
    }
    ln
}
impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let w_s= self.work_space.iter().fold(String::new(), |acc, x| acc + &x.lock().unwrap().clone().to_string() + " ");
        let r_s = self.record_space.iter().fold(String::new(), |acc, x| acc + &x.lock().unwrap().clone().to_string() + " ");
        let ed=self.map_space.iter().map(
            |x|x.iter().map(
                |s| s.lock().unwrap().clone().to_string()).collect::<Vec<String>>().join(" "))
            .collect::<Vec<String>>();
        let edg=(0..ed.len()).map(|x| x.to_string()+": "+&ed[x]).collect::<Vec<String>>().join("\n");

        let ls=self.live_stack.iter().map(
            |x| x.lock().unwrap().to_string()
        ).collect::<Vec<String>>().join("...");
        write!(f,"{}\n {} \n {} \n {}",w_s,r_s,edg,ls)
    }
}

impl Graph{
    fn new(node:Vec<i64>,mut edges:HashMap<i64,Vec<i64>>) -> Self{
        if node.len() !=edges.iter().len(){
            panic!("there must be the same number of nodes as node edges");
        }
        let mut nodes: Vec<i64>=vec![0];
        nodes.extend(node);
        edges.insert(0,vec![]);
        let w_s =(0..nodes.len()).map(
            |x| Arc::new(Mutex::new(nodes[x] as i64))
        ).collect::<Vec<Arc<Mutex<i64>>>>();
        let r_s =(0..nodes.len()).map(
            |x| Arc::new(Mutex::new(nodes[x] as i64))
        ).collect::<Vec<Arc<Mutex<i64>>>>();
        //let most_connections=edges.values().max_by_key(|v| v.len()).unwrap().len();
        //println!("{} {:?} {:?} {:?}",most_connections, w_s, r_s, edges);
        let m_s: Vec<Vec<Arc<Mutex<i64>>>>=
            nodes.iter()
                .map(|node|
                    edges[node].iter()
                        .map(|connection| w_s[connection.abs() as usize].clone())
                        .collect::<Vec<Arc<Mutex<i64>>>>())
                .collect::<Vec<Vec<Arc<Mutex<i64>>>>>();

        Self{ work_space: w_s, record_space: r_s, map_space:m_s, live_stack:vec![]}
    }

    fn from_json(path: &str) -> Self{
        let file = File::open(path).expect("file not found");
        let json: serde_json::Value = serde_json::from_reader(file).expect("file should be proper JSON");
        let mut noderes=Vec::<i64>::new();
        let nodes = json["Node_list"].as_array().unwrap();
        let edges = json["Edge_list"].as_array().unwrap();
        let edgeres: Vec<[i64; 2]>= edges
            .iter()
            .map(|edge| {
                edge.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_i64().unwrap() as i64)
                    .collect::<Vec<i64>>()
                    .try_into()
                    .expect("Each edge must have exactly 2 elements")
            })
            .collect();
        let mut edg_final:HashMap<i64,Vec<i64>>=HashMap::new();

        //println!("{:?} {:?}",edges,nodes);
        let mut is_router:bool;
        for node in nodes {
            is_router=json["Which_are_router"][node.as_str().unwrap()].as_bool().unwrap();

            noderes.push(node.as_str().unwrap().parse::<i64>().unwrap()*(1+(-2)*is_router as i64));
        }
        for edge in edgeres {
            let is_a_router=json["Which_are_router"][(edge[0] as u64).to_string()].as_bool().unwrap();
            let is_b_router=json["Which_are_router"][(edge[1] as u64).to_string()].as_bool().unwrap();
            edg_final.entry(edge[0]*(1+(-2)*is_a_router as i64)).or_insert_with(Vec::new).push(edge[1]*(1+(-2)*is_b_router as i64));
            edg_final.entry(edge[1]*(1+(-2)*is_b_router as i64)).or_insert_with(Vec::new).push(edge[0]*(1+(-2)*is_a_router as i64));
        }
        Graph::new( noderes, edg_final)
    }

    fn cull(&mut self){
        let mut found:bool = true;
        while found {
            found=false;
            for node in 1..self.work_space.len() {
                if true_len(&self.map_space[(*self.work_space[node].lock().unwrap()).abs() as usize]) == 1 {
                    let mut nd=self.work_space[node].lock().unwrap();
                    *nd=0;
                    let connection=self.map_space[node][0].lock().unwrap().abs() as usize;
                    let mut destination= self.record_space[connection].lock().unwrap();
                    *destination= *destination
                        *(1+(-2)*(
                            (*destination>0) as i64
                        ));

                    found=true;
                }
            }
        }
    }

    fn find_start(&self,start:&Arc<Mutex<i64>>)->Result<&Arc<Mutex<i64>>,&'static str>{
        let mut found:Vec<&Arc<Mutex<i64>>>= vec![];
        let mut small_live_stack:Vec<&Arc<Mutex<i64>>>=vec![start];
        let mut new: Vec<&Arc<Mutex<i64>>>;
        while small_live_stack.len()>0{
            new=vec![];
            for node in &small_live_stack {
                for connection in self.map_space[*node.lock().unwrap() as usize].iter() {
                    if *connection.lock().unwrap() < 0 {
                        return Ok(connection);
                    }
                }
                new.extend(
                    self.map_space[*node.lock().unwrap() as usize]
                        .iter().filter(|x| found.iter().filter(|k| Arc::ptr_eq(&k,&x)).collect::<Vec<_>>().len()==0).collect::<Vec<&Arc<Mutex<i64>>>>());
            }
            found.extend(small_live_stack);
            small_live_stack=new.clone();
        }
    Err("No start found")
    }

    fn graph_size_and_shape(&self, start:&Arc<Mutex<i64>>) -> (usize,Vec<&Arc<Mutex<i64>>>){
        let mut found:Vec<&Arc<Mutex<i64>>>= vec![start];
        let mut small_live_stack:Vec<&Arc<Mutex<i64>>>=vec![start];
        let mut new:Vec<&Arc<Mutex<i64>>>;
        let mut tmp:Vec<&Arc<Mutex<i64>>>;
        while &small_live_stack.len()>&0{
            new=vec![];
            tmp=vec![];
            for node in &small_live_stack {
                tmp.extend(
                self.map_space[(*node.lock().unwrap()).abs() as usize]
                    .iter().filter(|x| found.iter()
                    .filter(|k| Arc::ptr_eq(&k,&x))
                    .collect::<Vec<_>>().len()
                    ==0).filter(|x| x.lock().unwrap().abs()!=0)
                    .collect::<Vec<&Arc<Mutex<i64>>>>());
                for dst in &tmp {
                    if !new.iter().any(|x| Arc::ptr_eq(dst,x)) {
                        new.push(*dst);
                    }
                }
            }
        small_live_stack=new.clone();
        found.extend(new.clone());
        }

    return [found.len(),found];
    }

    fn graph_size(&self, start:&Arc<Mutex<i64>>) ->usize{
        let mut found:Vec<&Arc<Mutex<i64>>>= vec![start];
        let mut small_live_stack:Vec<&Arc<Mutex<i64>>>=vec![start];
        let mut new:Vec<&Arc<Mutex<i64>>>;
        let mut tmp:Vec<&Arc<Mutex<i64>>>;
        while &small_live_stack.len()>&0{
            new=vec![];
            tmp=vec![];
            for node in &small_live_stack {
                tmp.extend(
                    self.map_space[(*node.lock().unwrap()).abs() as usize]
                        .iter().filter(|x|
                        found.iter()
                            .filter(|k| Arc::ptr_eq(&k,&x)).
                            collect::<Vec<_>>().len()
                        ==0).filter(|x| x.lock().unwrap().abs()!=0)
                        .collect::<Vec<&Arc<Mutex<i64>>>>());
                for dst in &tmp {
                    if !new.iter().any(|x| Arc::ptr_eq(dst,x)) {
                        new.push(*dst);
                    }

                }
            }
            small_live_stack=new.clone();
            found.extend(new.clone());

            //println!("{:?} {:?}", found.iter().map(|x|*x.lock().unwrap()).collect::<Vec<i64>>(),
                    // small_live_stack.iter().map(|x|*x.lock().unwrap()).collect::<Vec<i64>>());
        }
        //println!("{:?}", found.iter().map(|x|*x.lock().unwrap()).collect::<Vec<i64>>());
        return found.len();

    }

    fn is_solved(&self,start: &Arc<Mutex<i64>>) -> bool{
        let start: Arc<Mutex<i64>>=self.find_start(&start).expect("start not found").clone();
        let goal_size: u32=self.graph_size(&start) as u32;
        //println!("goal_size: {}",goal_size);
        let mut found:Vec<&Arc<Mutex<i64>>>= vec![];
        let mut small_live_stack:Vec<&Arc<Mutex<i64>>>=vec![&start];
        let mut new: Vec<&Arc<Mutex<i64>>>;
        while &small_live_stack.len()>&0{
            new=vec![];
            for node in &small_live_stack {
                new.extend(
                    (self.map_space[node.lock().unwrap().abs() as usize]
                        .iter().filter(|x| found.iter()
                        .filter(|k| Arc::ptr_eq(&k, &x))
                        .collect::<Vec<_>>().len() == 0))
                        .filter(|x| x.lock().unwrap().abs()!=0)
                        .collect::<Vec<&Arc<Mutex<i64>>>>());
            }
            found.extend(new.clone());
            small_live_stack=new.clone().into_iter().filter(|x| *x.lock().unwrap()<0).collect::<Vec<&Arc<Mutex<i64>>>>();
        }
    //println!("{:?}", found.iter().map(|x|*x.lock().unwrap()).collect::<Vec<i64>>());
    found.len()==goal_size as usize
    }

    fn min_cut_set(&self,start:Arc<Mutex<i64>>)->Result<Arc<Mutex<i64>>,&'static str>{
        let size_shape=self.graph_size_and_shape(&start);
        let mut joined:Vec<Vec<&Arc<Mutex<i64>>>>=vec![
            size_shape.1
                .iter().map(|x| vec![x])
                .collect::<Vec<&Arc<Mutex<i64>>>>()
        ];
        let size=size_shape.0;
        for i in 0..self.graph_size(&start)-1 {
            let mut active_stack=vec![&start];
            while active_stack.len()<size-i{
                let mut choices:HashMap<&Arc<Mutex<i64>>,u32>=HashMap::new();//id: strength

            }
        }
        return Ok(start);
    }
}
fn main() {
    let mut test_gh: Graph= Graph::from_json("src/for_testing.json");
    println!("{}",test_gh);
    println!("{}",test_gh.graph_size(&test_gh.work_space[1]));
    println!("{}",test_gh.is_solved(&test_gh.work_space[1]));
    test_gh.cull();
    println!("{}",test_gh);
    println!("{}",test_gh.graph_size(&test_gh.work_space[1]));
    println!("{}",test_gh.is_solved(&test_gh.work_space[1]));


}
