import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;


public class User {

    public static final int DEFAULT_MAX_DAYS = 360;
    public static final double DEFAULT_LAMBDA = 0.95;

    private final String name;
    private final CategoryRegistry categoryRegistry;
    private final ArrayList<Event> history = new ArrayList<>();

    private final double lambda;
    private final int maxDays;
    private final LocalDate referenceDate;

    private MyVector rawVector;
    private MyVector userMyVector;

    public User(String name, CategoryRegistry categoryRegistry, LocalDate referenceDate) {
        this(name, categoryRegistry, referenceDate, DEFAULT_LAMBDA, DEFAULT_MAX_DAYS);
    }

    public User(String name, CategoryRegistry categoryRegistry,
                LocalDate referenceDate, double lambda, int maxDays) {
        if (referenceDate == null) {
            throw new IllegalArgumentException(
                    "referenceDate must be explicit; using the wall clock makes "
                            + "the profile non-reproducible");
        }
        if (lambda <= 0.0 || lambda > 1.0) {
            throw new IllegalArgumentException("lambda must be in (0, 1], got " + lambda);
        }
        this.name = name;
        this.categoryRegistry = categoryRegistry;
        this.referenceDate = referenceDate;
        this.lambda = lambda;
        this.maxDays = maxDays;
        this.rawVector = MyVector.zero(categoryRegistry.getDimension());
        this.userMyVector = MyVector.zero(categoryRegistry.getDimension());
    }

    public void addEvent(Event event) {
        this.history.add(event);
    }

    /** Half-life in days implied by the current lambda: ln(2) / -ln(lambda). */
    public double halfLifeDays() {
        return Math.log(2.0) / -Math.log(lambda);
    }

    /**
     * R_c = sum over events e in category c with 0 <= age(e) <= maxDays of
     *       weight(e) * lambda^age(e);   V = R / ||R||_2
     */
    public void calculateWithDecayAndDynamics() {
        int dimension = categoryRegistry.getDimension();
        double[] acc = new double[dimension];
        int used = 0, outOfWindow = 0, unknownCategory = 0;

        for (Event event : history) {
            long daysAgo = ChronoUnit.DAYS.between(event.getDate(), referenceDate);

            if (daysAgo < 0 || daysAgo > maxDays) {
                outOfWindow++;
                continue;
            }

            Integer idx = categoryRegistry.getCategoryIndex(event.getCategoryId());
            if (idx == null) {
                unknownCategory++;
                continue;
            }

            acc[idx] += event.getWatchTime() * Math.pow(lambda, daysAgo);
            used++;
        }

        List<Double> coords = new ArrayList<>(dimension);
        for (double v : acc) coords.add(v);

        this.rawVector = new MyVector(coords);
        double norm = rawVector.norm();
        this.userMyVector = norm > 0.0 ? rawVector.scale(1.0 / norm) : rawVector;

        System.out.printf(Locale.US,
                "profile[%s] ref=%s lambda=%.3f (half-life %.1f d) window=%dd "
                        + "| events used=%d, out-of-window=%d, unknown-category=%d, ||R||=%.2f%n",
                name, referenceDate, lambda, halfLifeDays(), maxDays,
                used, outOfWindow, unknownCategory, norm);
    }

    public void printUserVector() {
        List<Double> coords = userMyVector.getCoordinates();
        System.out.printf("%nNormalized interest vector [%s]:%n", name);
        for (int i = 0; i < coords.size(); i++) {
            if (coords.get(i) > 0.01) {
                System.out.printf(Locale.US, "  %-25s %.4f%n",
                        "[" + categoryRegistry.getCategoryNameByIndex(i) + "]",
                        coords.get(i));
            }
        }
    }

    public List<Map.Entry<String, Double>> getTopCategories(int topN) {
        List<Double> coords = userMyVector.getCoordinates();
        List<Map.Entry<String, Double>> scored = new ArrayList<>();
        for (int i = 0; i < coords.size(); i++) {
            scored.add(Map.entry(categoryRegistry.getCategoryNameByIndex(i), coords.get(i)));
        }
        return scored.stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(topN)
                .collect(Collectors.toList());
    }

    public String getName() { return name; }
    public ArrayList<Event> getHistory() { return history; }
    public MyVector getVector() { return userMyVector; }
    public MyVector getRawVector() { return rawVector; }
    public double getLambda() { return lambda; }
    public int getMaxDays() { return maxDays; }
    public LocalDate getReferenceDate() { return referenceDate; }
}