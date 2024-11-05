import main
import time


def at_least_one_label_in_text(labels, text):
    return any([str(label) in text for label in labels])


def _loop_documents(documents, rag_shortener, labels, pydantic_form):

    all_scores = {}
    for field in pydantic_form.__fields__:
        all_scores[field] = []

    # iterate through documents
    for key in documents:
        print("loading doc", key)
        start_time = time.time()
        paper_text = documents[key].lower()
        paper_labels = labels[key]

    
        print("--------- retrieving")
        rag_shortener.set_document(paper_text)
        for field in paper_labels:
            #print("")
            field_labels = paper_labels[field]

            # avoid counting small parts of words, and replace _ with space
            f = []
            for l in field_labels:
                l = str(l).replace("_", " ")
                if len(l)  < 5:
                    f.append(" "+l+" ")
                    f.append(" "+l+".")
                else:
                    f.append(l)
            field_labels = f


            if len(field_labels): # there are labels
                if at_least_one_label_in_text(field_labels, paper_text):
                    found_labels = []
                    other_labels = []
                    for l in field_labels:
                        if at_least_one_label_in_text([l], paper_text):
                            found_labels.append(l)
                        else:
                            other_labels.append(l)
                    print(found_labels)
                    print("\t\t\t\t", other_labels)

                    #print(field_labels)
                else:
                    print("\t\t\t\t", field_labels)

                    #nodes = rag_shortener.retriever.retrieve(
                    #        rag_shortener.retrieval_prompts[field]
                    #        )
                    #for node in nodes:
            #            #print(node.get_text()[:50])
            #            chunk = node.get_text()
            #            print(node.get_score(), "\tRELEVANT: "+str(field_labels) if at_least_one_label_in_text(field_labels, chunk) else "")
            #    else:
            #        print("no label in text")
            #else:
            #    print("no labels")

    quit()


        ## evaluate
        #if labels:
        #    scores = evaluation_fnc(paper_labels, filled_form, verbose=False)

        #    print("score:", scores)
        #    print("\n")
        #    for field in scores:
        #        all_scores[field].append(scores[field])
        #    all_times.append(time.time()-start_time)


    #if labels:
    #    #print("________printing final scores:")
    #    #pprint.pprint(all_scores)
    #    means_by_field = {}
    #    for field in all_scores:
    #        print(field, np.mean(all_scores[field]))
    #        means_by_field[field] = np.mean(all_scores[field])

    #    # calculate mean score
    #    final_score = []
    #    final_accuracy = []
    #    final_similarity = []
    #    for field in all_scores:
    #        scores = all_scores[field]
    #        final_score.extend(scores)

    #        field_properties = form_filler.pydantic_form.schema()["properties"][field]
    #        if (
    #                field_properties["type"] == "integer" or
    #                (field_properties["type"] == "string" and "enum" in field_properties)
    #                ):
    #            print(field, " -- accuacy")
    #            final_accuracy.extend(scores)
    #        else:
    #            final_similarity.extend(scores)
    #            print(field, " -- similarity ")
    #        #print(field_properties["type"], "enum" in field_properties, field_properties)
    #    print(final_score)
    #    print(np.mean(final_score))
    #    
    #    info_to_log = means_by_field
    #    info_to_log["total_score"] = np.mean(final_score)
    #    info_to_log["total_accuracy"] = np.mean(final_accuracy)
    #    info_to_log["total_similarity"] = np.mean(final_similarity)
    #    info_to_log["seconds"] = np.mean(all_times)
    #    info_to_log["papers_skipped"] = skips

    #    return info_to_log






def load_modules(args):
    """
    Load using the loader in main.py for simplicity
    """

    args.ff_model = "None" # skip loading form filling llm
    args.similarity_k = 100
    prepared_kwargs = main.load_modules(args)

    prepared_kwargs["pydantic_form"] = prepared_kwargs["context_shortener"].pydantic_form
    prepared_kwargs["rag_shortener"] = prepared_kwargs["context_shortener"]
    del prepared_kwargs["context_shortener"]
    del prepared_kwargs["remove_fields"]
    del prepared_kwargs["form_filler"]
    del prepared_kwargs["evaluation_fnc"]
    return prepared_kwargs



if __name__ == "__main__":
    args = main.parse_terminal_arguments()
    prepared_kwargs = load_modules(args)
    _loop_documents(**prepared_kwargs)
