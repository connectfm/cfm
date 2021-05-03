package com.amplifyframework.datastore.generated.model;


import androidx.core.util.ObjectsCompat;

import java.util.Objects;
import java.util.List;

/** This is an auto generated class representing the Location type in your schema. */
public final class Location {
  private final Double lat;
  private final Double longi;
  private final Double radius;
  public Double getLat() {
      return lat;
  }
  
  public Double getLongi() {
      return longi;
  }
  
  public Double getRadius() {
      return radius;
  }
  
  private Location(Double lat, Double longi, Double radius) {
    this.lat = lat;
    this.longi = longi;
    this.radius = radius;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      Location location = (Location) obj;
      return ObjectsCompat.equals(getLat(), location.getLat()) &&
              ObjectsCompat.equals(getLongi(), location.getLongi()) &&
              ObjectsCompat.equals(getRadius(), location.getRadius());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getLat())
      .append(getLongi())
      .append(getRadius())
      .toString()
      .hashCode();
  }
  
  public static LatStep builder() {
      return new Builder();
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(lat,
      longi,
      radius);
  }
  public interface LatStep {
    LongiStep lat(Double lat);
  }
  

  public interface LongiStep {
    BuildStep longi(Double longi);
  }
  

  public interface BuildStep {
    Location build();
    BuildStep radius(Double radius);
  }
  

  public static class Builder implements LatStep, LongiStep, BuildStep {
    private Double lat;
    private Double longi;
    private Double radius;
    @Override
     public Location build() {
        
        return new Location(
          lat,
          longi,
          radius);
    }
    
    @Override
     public LongiStep lat(Double lat) {
        Objects.requireNonNull(lat);
        this.lat = lat;
        return this;
    }
    
    @Override
     public BuildStep longi(Double longi) {
        Objects.requireNonNull(longi);
        this.longi = longi;
        return this;
    }
    
    @Override
     public BuildStep radius(Double radius) {
        this.radius = radius;
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(Double lat, Double longi, Double radius) {
      super.lat(lat)
        .longi(longi)
        .radius(radius);
    }
    
    @Override
     public CopyOfBuilder lat(Double lat) {
      return (CopyOfBuilder) super.lat(lat);
    }
    
    @Override
     public CopyOfBuilder longi(Double longi) {
      return (CopyOfBuilder) super.longi(longi);
    }
    
    @Override
     public CopyOfBuilder radius(Double radius) {
      return (CopyOfBuilder) super.radius(radius);
    }
  }
  
}
