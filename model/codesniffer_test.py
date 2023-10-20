import random
import sys
sys.path.append('../')
from data.dataset import CodeSnifferDataset
from model.modules.sniffer import CodeSnifferNetwork
import torch
from codesniffer_class import Sniffer

random.seed(90)
BATCH_SIZE = 16
WORKERS = 4
NUM_LABELS=8
PATH = "models/codeSniffer.pth"


def test1():
    dataset = CodeSnifferDataset('../data/filtered_data.csv', '../data/code_files')
    print("Instantiated dataset")
    ind = random.randrange(len(dataset))
    input_ids, attention_mask, labels = dataset[ind]
    print("Got one tensor")
    model = CodeSnifferNetwork(num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        print("One tensor: ")
        y_pred = model.forward(input_ids, attention_mask)
        y_pred = y_pred.squeeze(0).tolist()
        y_pred = [round(num, 2) for num in y_pred]
        y_pred = dict(zip(model.smells, y_pred))
        print(f"y_pred = {y_pred}")
        y_true = labels.tolist()
        y_true = dict(zip(model.smells, y_true))
        print(f"y_true = {y_true}")


def teste2(jcode):
    model = Sniffer(PATH)
    result = model.CodeAnalysis(jcode)
    print(result)


def main():
    jcode = """

        package org.apache.commons.jxpath;

        /**
         * A generic mechanism for accessing collections of name/value pairs.
         * Examples of such collections are HashMap, Properties,
         * ServletContext.  In order to add support for a new such collection
         * type to JXPath, perform the following two steps:
         * <ol>
         * <li>Build an implementation of the DynamicPropertyHandler interface
         * for the desired collection type.</li>
         * <li>Invoke the static method {@link JXPathIntrospector#registerDynamicClass
         * JXPathIntrospector.registerDynamicClass(class, handlerClass)}</li>
         * </ol>
         * JXPath allows access to dynamic properties using these three formats:
         * <ul>
         * <li><code>"myMap/myKey"</code></li>
         * <li><code>"myMap[@name = 'myKey']"</code></li>
         * <li><code>"myMap[name(.) = 'myKey']"</code></li>
         * </ul>
         */
        public interface DynamicPropertyHandler {
        
            /**
             * Returns a list of dynamic property names for the supplied object.
             * @param object to inspect
             * @return String[]
             */
            String[] getPropertyNames(Object object);

            /**
             * Returns the value of the specified dynamic property.
             * @param object to search
             * @param propertyName to retrieve
             * @return Object
             */
            Object getProperty(Object object, String propertyName);

            /**
             * Modifies the value of the specified dynamic property.
             * @param object to modify
             * @param propertyName to modify
             * @param value to set
             */
            void setProperty(Object object, String propertyName, Object value);
        }
    """
    teste2(jcode)

if __name__ == '__main__':
    main()