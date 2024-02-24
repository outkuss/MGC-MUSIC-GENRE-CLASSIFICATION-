import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;

public class MusicGenreClassifier {

    public static void main(String[] args) {
        try {
            // Veri setini yükle
            DataSource source = new DataSource("music_dataset.arff");
            Instances dataset = source.getDataSet();
            
            // Hedef sınıf (müzik türü) sütununu belirt
            dataset.setClassIndex(dataset.numAttributes() - 1);
            
            // Sınıflandırıcıyı oluştur
            Classifier classifier = new MultilayerPerceptron();
            
            // Sınıflandırıcıyı eğit
            classifier.buildClassifier(dataset);
            
            // Sınıflandırıcıyı değerlendir
            Evaluation eval = new Evaluation(dataset);
            eval.evaluateModel(classifier, dataset);
            System.out.println(eval.toSummaryString("\nResults\n======\n", false));
            
            // Örnek bir veri üzerinde sınıflandırma yap
            Instance newMusic = new DenseInstance(dataset.numAttributes());
            newMusic.setDataset(dataset);
            newMusic.setValue(0, 120); // Örnek bir müzik özelliği
            newMusic.setValue(1, 4);   // Örnek bir müzik özelliği
            newMusic.setValue(2, 300); // Örnek bir müzik özelliği
            newMusic.setValue(3, 1);   // Örnek bir müzik özelliği
            
            double predictedClass = classifier.classifyInstance(newMusic);
            System.out.println("Predicted music genre: " + dataset.classAttribute().value((int) predictedClass));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}

