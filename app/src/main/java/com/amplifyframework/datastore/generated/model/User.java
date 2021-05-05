package com.amplifyframework.datastore.generated.model;


import java.util.List;
import java.util.UUID;
import java.util.Objects;

import androidx.core.util.ObjectsCompat;

import com.amplifyframework.core.model.AuthStrategy;
import com.amplifyframework.core.model.Model;
import com.amplifyframework.core.model.ModelOperation;
import com.amplifyframework.core.model.annotations.AuthRule;
import com.amplifyframework.core.model.annotations.Index;
import com.amplifyframework.core.model.annotations.ModelConfig;
import com.amplifyframework.core.model.annotations.ModelField;
import com.amplifyframework.core.model.query.predicate.QueryField;

import static com.amplifyframework.core.model.query.predicate.QueryField.field;

/** This is an auto generated class representing the User type in your schema. */
@SuppressWarnings("all")
@ModelConfig(pluralName = "Users", authRules = {
  @AuthRule(allow = AuthStrategy.PUBLIC, operations = { ModelOperation.CREATE, ModelOperation.UPDATE, ModelOperation.DELETE, ModelOperation.READ })
})
public final class User implements Model {
  public static final QueryField ID = field("id");
  public static final QueryField EMAIL = field("email");
  public static final QueryField TASTE = field("taste");
  public static final QueryField BIAS = field("bias");
  public static final QueryField LOCATION = field("location");
  public static final QueryField HISTORY = field("history");
  private final @ModelField(targetType="ID", isRequired = true) String id;
  private final @ModelField(targetType="String", isRequired = true) String email;
  private final @ModelField(targetType="SongFeatures") SongFeatures taste;
  private final @ModelField(targetType="Float") Double bias;
  private final @ModelField(targetType="Location") Location location;
  private final @ModelField(targetType="History") History history;
  public String getId() {
      return id;
  }
  
  public String getEmail() {
      return email;
  }
  
  public SongFeatures getTaste() {
      return taste;
  }
  
  public Double getBias() {
      return bias;
  }
  
  public Location getLocation() {
      return location;
  }
  
  public History getHistory() {
      return history;
  }
  
  private User(String id, String email, SongFeatures taste, Double bias, Location location, History history) {
    this.id = id;
    this.email = email;
    this.taste = taste;
    this.bias = bias;
    this.location = location;
    this.history = history;
  }
  
  @Override
   public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      } else if(obj == null || getClass() != obj.getClass()) {
        return false;
      } else {
      User user = (User) obj;
      return ObjectsCompat.equals(getId(), user.getId()) &&
              ObjectsCompat.equals(getEmail(), user.getEmail()) &&
              ObjectsCompat.equals(getTaste(), user.getTaste()) &&
              ObjectsCompat.equals(getBias(), user.getBias()) &&
              ObjectsCompat.equals(getLocation(), user.getLocation()) &&
              ObjectsCompat.equals(getHistory(), user.getHistory());
      }
  }
  
  @Override
   public int hashCode() {
    return new StringBuilder()
      .append(getId())
      .append(getEmail())
      .append(getTaste())
      .append(getBias())
      .append(getLocation())
      .append(getHistory())
      .toString()
      .hashCode();
  }
  
  @Override
   public String toString() {
    return new StringBuilder()
      .append("User {")
      .append("id=" + String.valueOf(getId()) + ", ")
      .append("email=" + String.valueOf(getEmail()) + ", ")
      .append("taste=" + String.valueOf(getTaste()) + ", ")
      .append("bias=" + String.valueOf(getBias()) + ", ")
      .append("location=" + String.valueOf(getLocation()) + ", ")
      .append("history=" + String.valueOf(getHistory()))
      .append("}")
      .toString();
  }
  
  public static EmailStep builder() {
      return new Builder();
  }
  
  /** 
   * WARNING: This method should not be used to build an instance of this object for a CREATE mutation.
   * This is a convenience method to return an instance of the object with only its ID populated
   * to be used in the context of a parameter in a delete mutation or referencing a foreign key
   * in a relationship.
   * @param id the id of the existing item this instance will represent
   * @return an instance of this model with only ID populated
   * @throws IllegalArgumentException Checks that ID is in the proper format
   */
  public static User justId(String id) {
    try {
      UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
    } catch (Exception exception) {
      throw new IllegalArgumentException(
              "Model IDs must be unique in the format of UUID. This method is for creating instances " +
              "of an existing object with only its ID field for sending as a mutation parameter. When " +
              "creating a new object, use the standard builder method and leave the ID field blank."
      );
    }
    return new User(
      id,
      null,
      null,
      null,
      null,
      null
    );
  }
  
  public CopyOfBuilder copyOfBuilder() {
    return new CopyOfBuilder(id,
      email,
      taste,
      bias,
      location,
      history);
  }
  public interface EmailStep {
    BuildStep email(String email);
  }
  

  public interface BuildStep {
    User build();
    BuildStep id(String id) throws IllegalArgumentException;
    BuildStep taste(SongFeatures taste);
    BuildStep bias(Double bias);
    BuildStep location(Location location);
    BuildStep history(History history);
  }
  

  public static class Builder implements EmailStep, BuildStep {
    private String id;
    private String email;
    private SongFeatures taste;
    private Double bias;
    private Location location;
    private History history;
    @Override
     public User build() {
        String id = this.id != null ? this.id : UUID.randomUUID().toString();
        
        return new User(
          id,
          email,
          taste,
          bias,
          location,
          history);
    }
    
    @Override
     public BuildStep email(String email) {
        Objects.requireNonNull(email);
        this.email = email;
        return this;
    }
    
    @Override
     public BuildStep taste(SongFeatures taste) {
        this.taste = taste;
        return this;
    }
    
    @Override
     public BuildStep bias(Double bias) {
        this.bias = bias;
        return this;
    }
    
    @Override
     public BuildStep location(Location location) {
        this.location = location;
        return this;
    }
    
    @Override
     public BuildStep history(History history) {
        this.history = history;
        return this;
    }
    
    /** 
     * WARNING: Do not set ID when creating a new object. Leave this blank and one will be auto generated for you.
     * This should only be set when referring to an already existing object.
     * @param id id
     * @return Current Builder instance, for fluent method chaining
     * @throws IllegalArgumentException Checks that ID is in the proper format
     */
    public BuildStep id(String id) throws IllegalArgumentException {
        this.id = id;
        
        try {
            UUID.fromString(id); // Check that ID is in the UUID format - if not an exception is thrown
        } catch (Exception exception) {
          throw new IllegalArgumentException("Model IDs must be unique in the format of UUID.",
                    exception);
        }
        
        return this;
    }
  }
  

  public final class CopyOfBuilder extends Builder {
    private CopyOfBuilder(String id, String email, SongFeatures taste, Double bias, Location location, History history) {
      super.id(id);
      super.email(email)
        .taste(taste)
        .bias(bias)
        .location(location)
        .history(history);
    }
    
    @Override
     public CopyOfBuilder email(String email) {
      return (CopyOfBuilder) super.email(email);
    }
    
    @Override
     public CopyOfBuilder taste(SongFeatures taste) {
      return (CopyOfBuilder) super.taste(taste);
    }
    
    @Override
     public CopyOfBuilder bias(Double bias) {
      return (CopyOfBuilder) super.bias(bias);
    }
    
    @Override
     public CopyOfBuilder location(Location location) {
      return (CopyOfBuilder) super.location(location);
    }
    
    @Override
     public CopyOfBuilder history(History history) {
      return (CopyOfBuilder) super.history(history);
    }
  }
  
}
