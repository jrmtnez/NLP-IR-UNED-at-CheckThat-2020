import matplotlib.pyplot as plt
import networkx as nx
import gzip
import codecs
import json
import random

class TweetsGraph():

    def __init__(self, tweets_path, build_all=True, frac=1):
        self.tweets_array = []        
        # with gzip.open(tweets_path, "rt", encoding="utf-8") as tweets:
        with open(tweets_path, "rt", encoding="utf-8") as tweets:
            for tweet in tweets:
                decoded_data = codecs.decode(tweet.encode(), "utf-8")
                tweet_json = json.loads(decoded_data)
                self.tweets_array.append(tweet_json)
        if frac < 1:
            self.sample_tweets(frac=frac)
        if build_all:                        
            self.extract_triples()
            self.build_graph()


    def sample_tweets(self, frac=1):
        self.tweets_array = random.sample(self.tweets_array, int(len(self.tweets_array) * frac))
    
    def extract_triples(self):
        self.triples = []
        for tweet in self.tweets_array:            

            triple = (tweet["id"], "tweeted_by", tweet["user"]["screen_name"])
            self.triples.append(triple)

            try:
                triple = (tweet["id"], "quoted", tweet["quoted_status_id"])
                self.triples.append(triple)
            except:
                pass
            if tweet["in_reply_to_status_id"] is not None:
                triple = (tweet["id"], "reply_status", tweet["in_reply_to_status_id"])
                if tweet["id"] != tweet["in_reply_to_status_id"]:
                    self.triples.append(triple)
            if tweet["in_reply_to_screen_name"] is not None:
                triple = (tweet["user"]["screen_name"], "reply_user", tweet["in_reply_to_screen_name"])
                if tweet["user"]["screen_name"] != tweet["in_reply_to_screen_name"]:
                    self.triples.append(triple)
            for element in tweet:
                if element == "entities":
                    if tweet[element] is not None:
                        for level2_element in tweet[element]:
                            if level2_element == "hashtags":
                                for level3_element in tweet[element][level2_element]:
                                    triple = (tweet["id"], "hashtags", level3_element["text"])
                                    self.triples.append(triple)
                            if level2_element == "user_mentions":
                                for level3_element in tweet[element][level2_element]:
                                    triple = (tweet["user"]["screen_name"], "user_mentions", level3_element["screen_name"])
                                    self.triples.append(triple)
                                    triple = (tweet["id"], "has_mention", level3_element["screen_name"])
                                    self.triples.append(triple)
                            if level2_element == "urls":
                                for level3_element in tweet[element][level2_element]:
                                    triple = (tweet["id"], "contain_url", level3_element["display_url"])
                                    self.triples.append(triple)

    def build_graph(self):
        self.graph = nx.MultiDiGraph()
        for triple in self.triples:
            if triple[1] == "tweeted_by":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "user")
            if triple[1] == "quoted":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "tweet")
            if triple[1] == "reply_status":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "tweet")
            if triple[1] == "reply_user":
                self.graph.add_node(triple[0], label = "user")
                self.graph.add_node(triple[2], label = "user")
            if triple[1] == "hashtags":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "hashtag")
            if triple[1] == "user_mentions":
                self.graph.add_node(triple[0], label = "user")
                self.graph.add_node(triple[2], label = "user")
            if triple[1] == "has_mention":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "user")
            if triple[1] == "contain_url":
                self.graph.add_node(triple[0], label = "tweet")
                self.graph.add_node(triple[2], label = "url")
            self.graph.add_edge(triple[0], triple[2], label=triple[1])
        
        self.rev_graph = nx.reverse_view(self.graph)

    def get_node_attributes(self):
        return nx.get_node_attributes(self.graph, "label")
    
    def get_feature_nodes(self, node):
        feature_nodes = []
        rel_nodes, inner_nodes = self.get_related_nodes(node, "hashtags")
        for i in range(len(inner_nodes)):
            if len(feature_nodes) == 0:
                feature_nodes.append(rel_nodes[i][0])
        rel_nodes, inner_nodes = self.get_related_nodes(node, "quoted")
        for i in range(len(inner_nodes)):
            if len(feature_nodes) < 2:
                feature_nodes.append(rel_nodes[i][0])
        rel_nodes, inner_nodes = self.get_related_nodes(node, "reply_status")
        for i in range(len(inner_nodes)):
            if len(feature_nodes) < 3:
                feature_nodes.append(rel_nodes[i][0])
        rel_nodes, inner_nodes = self.get_related_nodes(node, "contain_url")
        for i in range(len(inner_nodes)):
            if len(feature_nodes) < 3:
                feature_nodes.append(rel_nodes[i][0])
        rel_nodes, inner_nodes = self.get_related_nodes(node, "has_mention")
        for i in range(len(inner_nodes)):
            if len(feature_nodes) < 3:
                feature_nodes.append(rel_nodes[i][0])   
        return feature_nodes

    def get_related_nodes(self, node, neighbor_type):
        inner_nodes_list = []
        rel_nodes_list = []
        inner_nodes = self.get_node_neighbors(self.graph, None, node, neighbor_type)
        for inner_node in inner_nodes:
            rel_nodes = self.get_node_neighbors(self.rev_graph, node, inner_node, neighbor_type)
            if len(rel_nodes) > 0:
                inner_nodes_list.append(inner_node)
                rel_nodes_list.append(rel_nodes)
        return rel_nodes_list, inner_nodes_list
    
    def get_node_neighbors(self, graph, parent_node, node, neighbor_type):
        node_list = []
        for neighbor in graph[node]:
            if graph[node][neighbor][0]['label'] == neighbor_type:
                if parent_node != neighbor:
                    node_list.append(neighbor)
        return node_list

    def plot(self, layout="spring", figsize=(300, 200), dpi=80):
        node_deg = nx.degree(self.graph)
        if layout == "spring":
            layout = nx.spring_layout(self.graph, k=1, iterations=100)
        if layout == "planar":
            layout = nx.planar_layout(self.graph, scale=2, dim=2)
        if layout == "shell":
            layout = nx.shell_layout(self.graph)
        plt.figure(num=None, figsize=figsize, dpi=dpi)
        nx.draw_networkx(
            self.graph,
            node_size=[int(deg[1]) * 500 for deg in node_deg],
            arrowsize=20,
            linewidths=1.5,
            pos=layout,
            edge_color='red',
            edgecolors='black',
            node_color='white')
        nx.draw_networkx_edge_labels(self.graph, pos=layout, edge_labels=None, font_color='blue', rotate=False)
        plt.axis('off')
        plt.show()