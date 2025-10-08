use std::arch::x86_64;
use std::collections::HashMap;
use std::env::consts::ARCH;
use std::io::Write;
use std::{fmt, vec};
use std::fs::File;
//performance note i will be using Arc<Mutex<T>> over Rc<RefCell<T>>
// despite a performance cost that this will occur it will reduce the code overhaul
// i would have to do if i later decide to thread the process
use std::sync::{Arc, Mutex};

use serde_json::to_string;


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
fn get_true_first(nodes: &Vec<Arc<Mutex<i64>>>) -> Option<&Arc<Mutex<i64>>> {
    for node in nodes {
        if *node.lock().unwrap() != 0 {
            return Some(node);
        }
    }
    None
}
fn vec_vec_arc_mutex_to_string(nodes: Vec<Vec<&Arc<Mutex<i64>>>>)->Option<String>{
    let mut st:Vec<String>=vec![];
    for joinedset in nodes{
        st.push("[".to_string());
        //println!("temp: {:?}",st);
        for subset in joinedset{
            //println!("temp:{:?}",subset.lock().unwrap().to_string());
            st.push(subset.lock().unwrap().to_string());
        }
        st.push("]".to_string());
    }
    //println!("midway translation{:?}", st);
    return Some(st.join(" "));
}
fn vec_vec_list2_arc_mutex_to_string(nodes: &Vec<Vec<[Arc<Mutex<i64>>;2]>>)->Option<String>{
    let mut st:Vec<String>=vec![];
    for set in nodes{
        st.push("[".to_string());
        for pair in set {
            st.push(pair[0].lock().unwrap().to_string());
            st.push(pair[1].lock().unwrap().to_string());
            st.push(",".to_string());
        }
        st.push("]".to_string());
    }
    return Some(st.join(" "));
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
        //println!("culling: {}", self);
        let mut found:bool = true;
        while found {
            found=false;
            for node in 1..self.work_space.len() {
                if true_len(&self.map_space[(*self.work_space[node].lock().unwrap()).abs() as usize]) == 1 {
                    //println!("{}",self);
                    if self.live_stack.iter().any(|x| Arc::ptr_eq(x, &self.work_space[node])) {
                        self.live_stack.remove(self.live_stack.iter().position(
                            |x| Arc::ptr_eq(x, &self.work_space[node])
                        ).expect("live stack should contain node"));
                        self.live_stack.push(get_true_first(&self.map_space[node]).unwrap().clone());
                    }
                    let mut nd=self.work_space[node].lock().unwrap();
                    *nd=0;
                    let connection=get_true_first(&self.map_space[node]).unwrap().lock().unwrap().abs() as usize;
                    let mut destination= self.record_space[connection].lock().unwrap();
                    *destination= *destination
                        *(1+(-2)*(
                            (*destination>0) as i64
                        ));
                    //println!("in place 1");
                    //println!("cull: {} -> {}", node, connection);
                   
                
                                    
                    found=true;
                }
            }

            //println!("cull loop done: {}\n {}", self,found);
        }
        
/* 
    for node in 1..self.work_space.len() {
            if true_len(&self.map_space[node]) == 0 {
                *self.work_space[node].lock().unwrap()=0;
                *self.record_space[node].lock().unwrap()= self.record_space[node].lock().unwrap().abs() as i64
                    *(1+(-2)*((*self.record_space[node].lock().unwrap()>0) as i64));
            }
        }*/
        //println!("cull done: {}", self);
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

    fn graph_size_and_shape(&self, start:&Arc<Mutex<i64>>) -> (usize,Vec<Arc<Mutex<i64>>>){
        let mut found:Vec<Arc<Mutex<i64>>>= vec![start.clone()];
        let mut small_live_stack:Vec<&Arc<Mutex<i64>>>=vec![start];
        let mut new:Vec<&Arc<Mutex<i64>>>;
        let mut tmp:Vec<&Arc<Mutex<i64>>>;
        while &small_live_stack.len()>&0{
            new=vec![];
            tmp=vec![];
            for node in &small_live_stack {
                tmp.extend(
                self.map_space[(*node.lock().unwrap()).abs() as usize]
                    .iter()
                    .filter(|x| found.iter()
                    .filter(|k| Arc::ptr_eq(*k,*x))
                    .count()
                    ==0).filter(|x| x.lock().unwrap().abs()!=0)
                    .collect::<Vec<&Arc<Mutex<i64>>>>());
                for dst in &tmp {
                    if !new.iter().any(|x| Arc::ptr_eq(dst,x)) {
                        new.push(*dst);
                    }
                }
            }
        small_live_stack=new.clone();
        found.extend(new.iter().map(|x| (*x).clone()));

        }

    return (found.len(),found);
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
        if true_len(&self.map_space[start.lock().unwrap().abs() as usize]) == 0 {
            return true;
        }
        let start: Arc<Mutex<i64>>= match self.find_start(&start){
            Ok(x) => x.clone(),
            Err(_) => return false,

        };
        let goal_size: u32=self.graph_size(&start) as u32;
        println!("goal size: {}", goal_size);
        if goal_size<2{
            return true;
        }
        //println!("goal_size: {}",goal_size);
        println!("In is_solved: {}", self);
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
            println!("is solved find loop: " );
        }
    //println!("{:?}", found.iter().map(|x|*x.lock().unwrap()).collect::<Vec<i64>>());
    found.len()==goal_size as usize
    }
    
    fn min_cut_edge_cases(&self,size:&usize) -> Result<String,String>{
        if size < &2 {
            return Err(to_string("Graph is too small to cut").unwrap());
        }
        if  size== &2 {
            return Ok(to_string("R01").unwrap());// there are only two elements to return 0 and 1
        }
        else{
            return Ok(to_string("NE").unwrap());// no error
        }
    }
    
    fn min_cut_set(&self,start:Arc<Mutex<i64>>)->Result<Vec<Vec<[Arc<Mutex<i64>>;2]>>,&'static str>{
        let mut every_cut: Vec<([Vec<&Arc<Mutex<i64>>>;2], usize)>=Vec::new();
        let size_shape=self.graph_size_and_shape(&start);
        match self.min_cut_edge_cases(&size_shape.0) {
            Ok(val)=> if val=="R01" {
                return Ok(vec![vec![[size_shape.1[0].clone(),size_shape.1[1].clone()]]])},
            Err(_)=> return Err("Graph is too small to cut"),
        }
        //catches the main edge cases

        let mut joined: Vec<Vec<&Arc<Mutex<i64>>>> = size_shape.1
            .iter()
            .map(|x| vec![x]).collect();//as nodes become grouped they are automatically treated as lists of nodes
        // therefore a node in the joined set starts of as [[1],[2],[3]] but may end up as [[1,2],[3]] 
        let size=size_shape.0;
        let mut choices:HashMap<usize,u32>=HashMap::new();
        println!("len of the joined set {}\njoined set{:?}",joined.len(),vec_vec_arc_mutex_to_string(joined.clone()).unwrap());
        for i in 0..self.graph_size(&start)-1 {// once a loop the size of the joined set should decrease by one
            let mut active_stack=vec![joined.iter().position(
                |x| x.iter().any(
                    |x| Arc::ptr_eq(x, &start)))
                .unwrap()];// gets the index of the starting value, when it may possibly be withing a joined node

            while active_stack.len()<size-i{
                choices=HashMap::new();//index into joined: strength
                for node in &active_stack {

                    let connections: Vec<&Arc<Mutex<i64>>> = joined[*node]
                                    .iter()
                                    .flat_map(|joined_nodes: &&Arc<Mutex<i64>>| {
                                            let idx = (*joined_nodes.lock().unwrap()).abs() as usize;
                                            self.map_space[idx]
                                                .iter()
                                                .filter(|x: &&Arc<Mutex<i64>>| x.lock().unwrap().abs() != 0)
                                                .collect::<Vec<&Arc<Mutex<i64>>>>()
                                        }).collect::<Vec<&Arc<Mutex<i64>>>>();
                                
                    for connection in connections {
                        let index=joined.iter().position(|x| 
                            x.iter().any(
                                |a| 
                                Arc::ptr_eq( *a, connection)) )
                            .unwrap();
                        *choices.entry(index).or_insert(0)+=1;
                        }
                    
                    }
                    let choice= choices.iter().max_by_key(|x| x.1).unwrap().0;
                    
                    active_stack.push(*choice);
                }
            // this adds one element in a chain one after another, according to the stoer wagner algorithm.

            let last_pushed_idex=active_stack[active_stack.len()-1];
            let stren = choices[&last_pushed_idex];
            every_cut.push(
                        ([
                            joined[active_stack[active_stack.len() - 1]]
                                .iter()
                                .cloned()
                                .collect::<Vec<&Arc<Mutex<i64>>>>(),
                            joined[active_stack[active_stack.len() - 2]]
                                .iter()
                                .cloned()
                                .collect::<Vec<&Arc<Mutex<i64>>>>(),
                        ],
                        stren as usize)
                    );// the cut is added to the every cut list, the cut is held as 2 sets of nodes who's interconnected edges should be cut
                    // and the number of connections between them

            let a_idx = active_stack[active_stack.len() - 2];
            let b_idx = active_stack[active_stack.len() - 1];

            if a_idx != b_idx {
                let (first, second) = if a_idx < b_idx {
                    let (first, second) = joined.split_at_mut(b_idx);
                    (&mut first[a_idx], &second[0])
                } else {
                    let (first, second) = joined.split_at_mut(a_idx);
                    (&mut second[0], &first[b_idx])
                };

                first.extend(second.iter().cloned()); 
            }// this has something to do with joining the last two elements added to the stack, see the stoer wagner equation
            joined.remove(active_stack[active_stack.len() - 1]);
            println!("len of the joined set {}\njoined set{:?}",joined.len(),vec_vec_arc_mutex_to_string(joined.clone()).unwrap());
        
           }

        
        //in theory the section bellow is just error cleanup
        // it is making sure there are no cuts that reference the same nodes 
        let mini:usize = every_cut.iter().map(|x| x.1).min().unwrap();
        let tmp_min_cuts: Vec<Vec<[Arc<Mutex<i64>>;2]>> = every_cut.iter().filter(|x| x.1 == mini)
            .map(|x: &([Vec<&Arc<Mutex<i64>>>; 2], usize)| {
                let mut all_edges:Vec<[Arc<Mutex<i64>>;2]>=vec![];
                
                for node in x.0[0].iter(){
                    let linked: &Vec<Arc<Mutex<i64>>>=&self.map_space[ node.lock().unwrap().abs() as usize];
                    for destination in x.0[1].iter() {
                        if linked.iter().any(|des| Arc::ptr_eq(des, destination)) {
                            let mut edge: [Arc<Mutex<i64>>; 2] = [(*node).clone(), (*destination).clone()];
                            edge.sort_by(|a, b| a.lock().unwrap().cmp(&b.lock().unwrap()));
                            all_edges.push(edge);
                        }
                    }
                }
                all_edges
            }).collect();
            
        

        return Ok(tmp_min_cuts);
        
    }
    

    fn min_cut_set_2(&self,start:Arc<Mutex<i64>>)->Result<Vec<Vec<[Arc<Mutex<i64>>;2]>>,&'static str>{
        let size_shape=self.graph_size_and_shape(&start);
        match self.min_cut_edge_cases(&size_shape.0) {
            Ok(val)=> if val=="R01" {
                return Ok(vec![vec![[size_shape.1[0].clone(),size_shape.1[1].clone()]]])},
            Err(_)=> return Err("Graph is too small to cut"),
        }


        let mut joined: Vec<Vec<&Arc<Mutex<i64>>>> = size_shape.1
            .iter()
            .map(|x| vec![x]).collect();//as nodes become grouped they are automatically treated as lists of nodes
        // therefore a node in the joined set starts of as [[1],[2],[3]] but may end up as [[1,2],[3]] 
        let cuts;
        let mut len_joined:usize;
        for i in 0..size_shape.0{
            len_joined=joined.len();
            



            //end check that joined has reduced by one
            if joined.len()!=len_joined-1 && i>0{
                panic!("joined set did not reduce in size");
            }
        }

        return Err("pass");

    }

    fn find_last_step_stoer_wagner(&self,start:Arc<Mutex<i64>>,joined:&Vec<Vec<&Arc<Mutex<i64>>>>)->Result<Vec<u64>,&'static str>{
        
        let mut locally_joined:Vec<Vec<&Arc<Mutex<i64>>>>=joined.clone();
        let mut start_ind: u64= locally_joined.iter().position(
            |x| x.iter().any(
                |a| Arc::ptr_eq(a, &start))).unwrap() as u64;
        let start_ind_fixed:u64=start_ind.clone();
        let mut next_set:Vec<&Arc<Mutex<i64>>>; 

        while locally_joined.len()>2{
            let next=self.next_step_in_stoer(
                start_ind,
                &locally_joined[start_ind as usize],
                 &locally_joined);

            if next==start_ind{
                return Err("next step in stoer wagner returned the same index as the working set");
            } else if next>=locally_joined.len() as u64{
                return Err("next step in stoer wagner returned an index out of bounds");
            }

            next_set=locally_joined[next as usize].clone();
            locally_joined.remove(next as usize);
             if start_ind>next{
                start_ind-=1;
            }
            locally_joined[start_ind as usize].extend(next_set);
           
            
        }

        let other_one_element: &Arc<Mutex<i64>>= locally_joined.get(
            match start_ind {
                0=>1,
                1=>0,
                _=>panic!("The start index is somehow not in 0 or 1")}
            ).unwrap()[0];
        
        let other_ind= joined.iter().position(
            |x| x.iter().any(
                |a| Arc::ptr_eq(a, other_one_element)))
            .expect("The other index is not found it the origianl set");

        let second_to_last= joined.iter().position(
            |x| x.iter().any(
                |a| Arc::ptr_eq(a, next_set[0])));
        return Ok(vec![start_ind_fixed as u64, other_ind as u64]);
        
        

    }

    fn next_step_in_stoer(&self,ind_of_working_set: u64,working_set:&Vec<&Arc<Mutex<i64>>>, joined_set_local: &Vec<Vec<&Arc<Mutex<i64>>>>)-> u64{
        let strength_of_connection=self.count_locals(ind_of_working_set,working_set.clone(), joined_set_local);

        let best: &u64= strength_of_connection.iter().max_by(|a ,b | a.1.cmp(b.1)).unwrap().0;//remember to remove 0 that will show up a lot so it needs to not be counted
        return *best
        
    }   

    fn count_locals(&self,ind_of_working_set:u64, working_set:Vec<&Arc<Mutex<i64>>>,joined_set_local: &Vec<Vec<&Arc<Mutex<i64>>>>)-> HashMap<u64,u64>{ 
        let mut strength_of_connection: HashMap<u64,u64>=HashMap::new();
       
        for node in &working_set{

            for connection in &self.map_space[node.lock().unwrap().abs() as usize]{
                //for every node in the working set it needs to find the connection it has to 
                // every node in the local joined set. it must rember that one joined node may be
                // made up of many nodes
                
                let ind_of_joined= match  joined_set_local.iter().position(
                        |x| x.iter().any(
                            |a| Arc::ptr_eq(a, &connection)
                        )){
                             Some(x) => x as u64,
                             None => ind_of_working_set,
                        };  //if the connection is not in the joined set it must be in the working set or it is 0
                        //if it is 0 it will be ignored in the next step

                if ind_of_joined==ind_of_working_set{
                    continue;
                }
                else  {
                    *strength_of_connection.entry(ind_of_joined as u64).or_insert(0) += 1;
                }
            }
            
        }
        return strength_of_connection;
    }

    fn clear_section(&mut self, start: &Arc<Mutex<i64>>){
        let clear_section= self.graph_size_and_shape(start).1;
        for node in clear_section {
            let mut n=node.lock().unwrap();
            *n=0;
        }
    }
    
    fn score_pair(&self, pair: &[Arc<Mutex<i64>>; 2]) -> usize {
        self.map_space[pair[0].lock().unwrap().abs() as usize].len()+self.map_space[pair[1].lock().unwrap().abs() as usize].len()
    }

    fn best_pair(&mut self, cuts: &Vec<[Arc<Mutex<i64>>;2]>) -> [Arc<Mutex<i64>>;2] {
        let mut best_pair:[&Arc<Mutex<i64>>;2] = [&cuts[0][0], &cuts[0][1]];
        let mut best_score:usize = 0;
        for cut in cuts {
            if self.score_pair(&[cut[0].clone(), cut[1].clone()])  > best_score {
                best_pair = [&cut[0], &cut[1]];
                best_score = self.score_pair(&[cut[0].clone(), cut[1].clone()]);
                // need to add the preference that less routers that are made the better
                // therefore if one of the pair is already a router that is better
            }
        }
        let result: [Arc<Mutex<i64>>; 2] = best_pair.iter().map(|x| (*x).clone()).collect::<Vec<_>>().try_into().unwrap();

        result

    }
    
    fn make_pair_router(&mut self, pair: &[Arc<Mutex<i64>>; 2]) {
        let mut a = pair[0].lock().unwrap();
        let mut b = pair[1].lock().unwrap();
        if *a>0{
            *a = -(*a);
            *self.record_space[(*a).abs() as usize].lock().unwrap() = -(*a);
        }
        if *b>0{
            *b = -(*b);
            *self.record_space[(*b).abs() as usize].lock().unwrap() = -(*b);
        }
    }
    
    fn find_routers(&mut self) -> Vec<i64> {
        self.live_stack.push(self.work_space[1].clone());
        while self.work_space.iter().any(|x| *x.lock().unwrap() != 0) {
            if self.live_stack.len() == 0 {
                panic!("no live nodes found but workspace empty");
            }
            //println!("in place 1");
            self.cull();
            println!("After cull: {}", self);
            if self.is_solved(&self.live_stack[0]){
                println!("Graph is solved: True \n{}", self);
                    let st =self.live_stack.pop()
                    .expect("live stack should not be empty");
                    self.clear_section(&st);
                    continue;
            }

            println!("After solved: {}", self);
            if self.live_stack.len() != 0 {
               
            
            let mins= self.min_cut_set(self.live_stack[0].clone()).expect("something has gone wrong with min cut set");
            println!("Min cuts found: {:?}", vec_vec_list2_arc_mutex_to_string(&mins));
            for cut in mins.iter() {
                let best_pair = self.best_pair(cut);
                self.make_pair_router(&best_pair);
                self.live_stack.push(best_pair[0].clone());
                self.live_stack.push(best_pair[1].clone());
                for cu in cut.iter(){
                    self.map_space[cu[0].lock().unwrap().abs() as usize].retain(|x| !Arc::ptr_eq(x, &cu[1]));
                    self.map_space[cu[1].lock().unwrap().abs() as usize].retain(|x| !Arc::ptr_eq(x, &cu[0]));
                    

                }

                
            }

        }
    }




        self.record_space.iter().filter(|x|*x.lock().unwrap()<0)
            .map(|x| *x.lock().unwrap())
            .collect::<Vec<i64>>()
    }
    
    fn submit_result(&self, routers: &Vec<i64>,path: &str) {
        let mut file = File::create(path).expect("Unable to create file");
        for router in routers {
            file.write(
                    format!("{}\n",router.abs().to_string()).as_bytes())
                    .expect("could not write");
        }
    }
}

fn main() {
    /* 
    let mut test_gh: Graph= Graph::from_json("src/for_testing.json");
    println!("{}",test_gh);
    println!("{}",test_gh.graph_size(&test_gh.work_space[1]));
    println!("{}",test_gh.is_solved(&test_gh.work_space[1]));
    test_gh.cull();
    println!("{}",test_gh);
    println!("{}",test_gh.graph_size(&test_gh.work_space[1]));
    println!("{}",test_gh.is_solved(&test_gh.work_space[1]));
    let routers = test_gh.find_routers();
    println!("Routers found: {:?}", routers);
    test_gh.submit_result(&routers, "src/routers.txt");
    println!("Routers submitted to file.");
    */
    // Test cases for the Graph implementation
    // These tests will read from JSON files and print the graph details, size, solved status

    let mut test_gh1: Graph= Graph::from_json("src/t1.json");
    println!("Output graph before any modification {}",test_gh1);
    println!("Size of graph {}",test_gh1.graph_size(&test_gh1.work_space[1]));
    println!("check if solved:{}",test_gh1.is_solved(&test_gh1.work_space[1]));
    println!("routers: {:?}", test_gh1.find_routers());


    let mut test_gh2: Graph= Graph::from_json("src/t2.json");
    println!("Output graph before modification\n{}",test_gh2);
    println!("Size of graph: {}",test_gh2.graph_size(&test_gh2.work_space[1]));
    println!("check if solved:{}",test_gh2.is_solved(&test_gh2.work_space[1]));
    println!("routers: {:?}", test_gh2.find_routers());

    let mut test_gh3: Graph= Graph::from_json("src/t3.json");
    println!("Output pre mod grpah\n{}",test_gh3);
    println!("Size:{}",test_gh3.graph_size(&test_gh3.work_space[1]));
    println!("Is solved{}",test_gh3.is_solved(&test_gh3.work_space[1]));
    println!("routers: {:?}", test_gh3.find_routers());
    
    test_gh3.submit_result(&test_gh3.record_space.iter().filter(|x|*x.lock().unwrap()<0).map(|x| *x.lock().unwrap()).collect::<Vec<i64>>(), "src/routers.txt");
    println!("Routers submitted to file.");

}
