package recommender.Model;

import java.util.*;

public class CategoryRegistry {
    private final Map<String, String> idToName = new HashMap<>();
    private final Map<String, Integer> idToIndex = new HashMap<>();
    private final Map<String, MyVector> categoryBasis = new HashMap<>();

    public CategoryRegistry() {
        initializeCategories();
        buildIndexMapping();
        buildBasisVectors();
    }

    private void initializeCategories() {
        idToName.put("1", "Film & Animation");
        idToName.put("2", "Autos & Vehicles");
        idToName.put("10", "Music");
        idToName.put("15", "Pets & Animals");
        idToName.put("17", "Sports");
        idToName.put("18", "Short Movies");
        idToName.put("19", "Travel & Events");
        idToName.put("20", "Gaming");
        idToName.put("21", "Videoblogging");
        idToName.put("22", "People & Blogs");
        idToName.put("23", "Comedy");
        idToName.put("24", "Entertainment");
        idToName.put("25", "News & Politics");
        idToName.put("26", "Howto & Style");
        idToName.put("27", "Education");
        idToName.put("28", "Science & Technology");
        idToName.put("29", "Nonprofits & Activism");
    }

    private void buildIndexMapping() {
        int index = 0;
        for (String id : idToName.keySet()) {
            idToIndex.put(id, index++);
        }
    }

    private void buildBasisVectors() {
        int dimension = idToName.size();

        for (Map.Entry<String, Integer> entry : idToIndex.entrySet()) {
            String categoryId = entry.getKey();
            int index = entry.getValue();
            String categoryName = idToName.get(categoryId);

            ArrayList<Double> coords = new ArrayList<>(Collections.nCopies(dimension, 0.0));
            coords.set(index, 1.0);
            categoryBasis.put(categoryName, new MyVector(coords));
        }
    }

    public String getCategoryName(String id) {
        return idToName.get(id);
    }

    public Integer getCategoryIndex(String id) {
        return idToIndex.get(id);
    }

    public MyVector getBasisVector(String categoryName) {
        return categoryBasis.get(categoryName);
    }

    public Map<String, MyVector> getAllBasisVectors() {
        return categoryBasis;
    }

    public int getDimension() {
        return idToName.size();
    }

    public boolean isValidCategory(String id) {
        return idToName.containsKey(id);
    }

    public String getCategoryNameByIndex(int index) {
        for (Map.Entry<String, Integer> entry : idToIndex.entrySet()) {
            if (entry.getValue() == index) {
                String categoryId = entry.getKey();
                return idToName.get(categoryId);
            }
        }
        return "Unknown";
    }
}
