import java.util.*;

public class CategoryRegistry {

    private static final String[][] CATEGORIES = {
            {"1",  "Film & Animation"},
            {"2",  "Autos & Vehicles"},
            {"10", "Music"},
            {"15", "Pets & Animals"},
            {"17", "Sports"},
            {"18", "Short Movies"},
            {"19", "Travel & Events"},
            {"20", "Gaming"},
            {"21", "Videoblogging"},
            {"22", "People & Blogs"},
            {"23", "Comedy"},
            {"24", "Entertainment"},
            {"25", "News & Politics"},
            {"26", "Howto & Style"},
            {"27", "Education"},
            {"28", "Science & Technology"},
            {"29", "Nonprofits & Activism"},
    };

    private final Map<String, String> idToName = new LinkedHashMap<>();
    private final Map<String, Integer> idToIndex = new LinkedHashMap<>();
    private final Map<String, MyVector> categoryBasis = new LinkedHashMap<>();
    private final String[] indexToName;

    public CategoryRegistry() {
        for (String[] row : CATEGORIES) {
            idToName.put(row[0], row[1]);
        }
        int i = 0;
        indexToName = new String[idToName.size()];
        for (Map.Entry<String, String> e : idToName.entrySet()) {
            idToIndex.put(e.getKey(), i);
            indexToName[i] = e.getValue();
            i++;
        }
        buildBasisVectors();
    }

    private void buildBasisVectors() {
        int dimension = idToName.size();
        for (Map.Entry<String, Integer> entry : idToIndex.entrySet()) {
            List<Double> coords = new ArrayList<>(Collections.nCopies(dimension, 0.0));
            coords.set(entry.getValue(), 1.0);
            categoryBasis.put(idToName.get(entry.getKey()), new MyVector(coords));
        }
    }

    public String getCategoryName(String id) {
        return idToName.get(id);
    }

    public Integer getCategoryIndex(String id) {
        return idToIndex.get(id);
    }

    public String getCategoryNameByIndex(int index) {
        return (index >= 0 && index < indexToName.length) ? indexToName[index] : "Unknown";
    }

    public MyVector getBasisVector(String categoryName) {
        return categoryBasis.get(categoryName);
    }

    public Map<String, MyVector> getAllBasisVectors() {
        return Collections.unmodifiableMap(categoryBasis);
    }

    public int getDimension() {
        return idToName.size();
    }

    public boolean isValidCategory(String id) {
        return idToName.containsKey(id);
    }
}