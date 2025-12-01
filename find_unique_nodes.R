##code to find the unique nodes in the network, and also check in degree and out degree

source("libraries_etc.R")

####################################################################################

##load in the network

ds_network<-data.table(read.csv("pa_ds_network.csv"))

##calculate the in degree and out degree so we can check it with the data in the paper (check on page 81 of the paper)

out_degree<-ds_network[,.(out_degree=.N), by=From]
in_degree<-ds_network[,.(in_degree=.N), by=To]

names(out_degree)<-c("Node", "Out degree")
names(in_degree)<-c("Node", "In degree")

all_degrees<-merge(in_degree, out_degree, all = T)

all_degrees[is.na(all_degrees)]<-0

##NOTE: in-degree for family cohesion is one higher than in the publication, but this appears to be right













