package recommender.Model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Immutable clean class
 */
public final class MyVector {
    private final List<Double> coordinates;

    public MyVector(List<Double> coordinates) {
        if (coordinates == null || coordinates.isEmpty()) {
            throw new IllegalArgumentException("Vector cannot be empty");
        }
        // Full immutable
        this.coordinates = List.copyOf(coordinates);
    }

    public MyVector add(MyVector other) {
        if (other.size() != this.size()) {
            throw new IllegalArgumentException("Different sizes of vectors: " + this.size() + " != " + other.size());
        }

        List<Double> result = new ArrayList<>(this.size());
        for (int i = 0; i < this.size(); i++) {
            result.add(this.coordinates.get(i) + other.coordinates.get(i));
        }
        return new MyVector(result);
    }

    public MyVector scale(double factor) {
        List<Double> result = new ArrayList<>(this.size());
        for (double coord : this.coordinates) {
            result.add(coord * factor);
        }
        return new MyVector(result);
    }

    public double dot(MyVector other) {
        if (this.size() != other.size()) {
            throw new IllegalArgumentException("Different sizes of vectors");
        }

        double sum = 0;
        for (int i = 0; i < this.size(); i++) {
            sum += this.coordinates.get(i) * other.coordinates.get(i);
        }
        return sum;
    }

    public double norm() {
        double total = 0;
        for (double value : coordinates) {
            total += value * value;
        }
        return Math.sqrt(total);
    }

    public double cosine(MyVector other) {
        double normProduct = this.norm() * other.norm();
    
        if (normProduct == 0.0) {
            return 0.0;
        }
        return this.dot(other) / normProduct;
    }

    public static MyVector zero(int dimension) {
        return new MyVector(Collections.nCopies(dimension, 0.0));
    }

    public List<Double> getCoordinates() {
        return this.coordinates;
    }

    public int size() {
        return coordinates.size();
    }

    public double get(int index) {
        return coordinates.get(index);
    }

    @Override
    public String toString() {
        return coordinates.toString();
    }
}
