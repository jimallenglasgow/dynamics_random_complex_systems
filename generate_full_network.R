##code to generate the final network  

source("libraries_etc.R")

####################################################################################

##load in the original network

ds_network<-data.table(read.csv("pa_ds_network.csv"))

##and the additional nodes

additional_nodes<-data.table(read.csv("additional_nodes.csv"))

##and combine

full_network<-rbind(ds_network, additional_nodes)

##assign nodes a number rather than a name

all_nodes<-unique(c(full_network$From, full_network$To))

node_ids<-data.table(Node=all_nodes, ID=NA)

node_ids[, ID:=1:.N]

node_ids[Node=="PA", ID:=0]

##and assign them to the network

from_node_ids<-node_ids

names(from_node_ids)<-c("From", "From_ID")

to_node_ids<-node_ids

names(to_node_ids)<-c("To", "To_ID")

full_network_from_ID<-merge(full_network, from_node_ids)

full_network_ID<-merge(full_network_from_ID, to_node_ids, by="To")

##and now create the adjacency matrix

full_network_ID$To<-NULL

full_network_ID$From<-NULL

full_network_ID<-full_network_ID[!is.na(Sign)]

full_network_ID[Sign=="+", Sign:=1]
full_network_ID[Sign=="-", Sign:=-1]
full_network_ID[Sign=="0", Sign:=2]

##add some self loops so that we get a square adjacency matrix

for(sel_id in node_ids$ID){
  
  sel_id_df<-data.table(Sign=NA, From_ID=sel_id, To_ID=sel_id)
  
  full_network_ID<-rbind(full_network_ID, sel_id_df)
  
}

full_network_ID_wide<-full_network_ID %>% spread(key = "To_ID", value="Sign")

full_network_ID_wide$From_ID<-NULL

full_network_ID_wide[is.na(full_network_ID_wide)]<-0

##save it

write.csv(full_network_ID_wide, "PA_network.csv", row.names = F)










