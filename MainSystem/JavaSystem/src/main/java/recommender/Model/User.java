package recommender.Model;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

/**
 * User's math profile
 */
public class User {
    private static final int MAX_DAYS = 360;

    private final String name;
    private final CategoryRegistry categoryRegistry;
    private MyVector userMyVector;
    private final ArrayList<Event> history;
    private double lambda = 0.95; // Cut 5% everyday

    public User(String name, CategoryRegistry categoryRegistry) {
        this.name = name;
        this.categoryRegistry = categoryRegistry;
        this.userMyVector = MyVector.zero(categoryRegistry.getDimension());
        this.history = new ArrayList<>();
    }

    public void addEvent(Event event) {
        this.history.add(event);
    }

    /**
     * Count user's interest vector
     */
    public void calculateWithDecayAndDynamics() {
        LocalDate today = LocalDate.now();
        int dimension = categoryRegistry.getDimension();

        // Saving weights in array
        double[] accumulatedCoords = new double[dimension];

        for (Event event : history) {
            long daysAgo = ChronoUnit.DAYS.between(event.getDate(), today);

            if (daysAgo < 0 || daysAgo > MAX_DAYS) continue;

            // Count decay: lambda^daysAgo
            double timeDecayFactor = Math.pow(lambda, daysAgo);
            double effectiveWeight = event.getWatchTime() * timeDecayFactor;

            Integer categoryIndex = categoryRegistry.getCategoryIndex(event.getCategoryId());

            if (categoryIndex != null) {
                accumulatedCoords[categoryIndex] += effectiveWeight;
            }
        }

        // Massive -> Vector
        List<Double> coordinatesList = new ArrayList<>(dimension);
        for (double val : accumulatedCoords) {
            coordinatesList.add(val);
        }

        MyVector accumulatedVector = new MyVector(coordinatesList);

        // L2-normalize
        double vectorNorm = accumulatedVector.norm();
        if (vectorNorm > 0.0) {
            this.userMyVector = accumulatedVector.scale(1.0 / vectorNorm);
        } else {
            this.userMyVector = accumulatedVector;
        }
    }


    public void printUserVector() {
        if (userMyVector == null) {
            System.out.println("Vector not calculated yet.");
            return;
        }

        System.out.println("\n📊 Normalized vector [TargetUser]:");
        List<Double> coords = userMyVector.getCoordinates();

        for (int i = 0; i < coords.size(); i++) {
            double value = coords.get(i);
            if (value > 0.01) {
                String categoryName = categoryRegistry.getCategoryNameByIndex(i);
                System.out.printf("  %-25s (Component %2d): %.4f%n",
                        "[" + categoryName + "]", i, value);
            }
        }
    }

    /**
     * Searching Top-N categories
     */
    public List<Map.Entry<String, Double>> getTopCategories(int topN) {
        Map<String, MyVector> basisVectors = categoryRegistry.getAllBasisVectors();

        return basisVectors.entrySet().stream()
                .map(entry -> {
                    double cosineDistance = this.getVector().cosine(entry.getValue());
                    return Map.entry(entry.getKey(), cosineDistance);
                })
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topN)
                .collect(Collectors.toList());
    }

    public String getName() { return name; }
    public ArrayList<Event> getHistory() { return history; }
    public MyVector getVector() { return userMyVector; }
    public double getLambda() { return lambda; }
    public void setLambda(double lambda) { this.lambda = lambda; }
}
