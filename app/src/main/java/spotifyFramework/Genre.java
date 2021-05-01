package spotifyFramework;

public enum Genre {
	ROCK, POP, COUNTRY, RAP, CLASSICAL, JAZZ, WEIRD;

	@Override
	public String toString() {
		String name = name();
		StringBuilder sb = new StringBuilder("Category: ");
		for (int i = 0; i < name.length(); i++) {
			if (name.charAt(i) == '_') {
				sb.append(' ');
			} else {
				sb.append(name.charAt(i));
			}
		}
		return sb.toString();
	}
}

