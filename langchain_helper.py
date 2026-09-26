from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate


# Ollama local model
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)


def generate_restaurant_name_and_items(cuisine):

    # Chain 1: Generate restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=["cuisine"],
        template="""
        I want to open a restaurant for {cuisine} food.
        Suggest a fancy restaurant name.
        Return only the restaurant name.
        """
    )

    name_prompt = prompt_template_name.invoke({
        "cuisine": cuisine
    })

    restaurant_name = llm.invoke(name_prompt).content


    # Chain 2: Generate menu items
    prompt_template_items = PromptTemplate(
        input_variables=["restaurant_name"],
        template="""
        Suggest some menu items for {restaurant_name}.
        Return the menu items as a comma separated string.
        """
    )

    items_prompt = prompt_template_items.invoke({
        "restaurant_name": restaurant_name
    })

    menu_items = llm.invoke(items_prompt).content


    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }


if __name__ == "__main__":
    response = generate_restaurant_name_and_items("Italian")

    print("Restaurant Name:")
    print(response["restaurant_name"])

    print("\nMenu Items:")
    print(response["menu_items"])